import argparse
import json
import time
import math
import numpy as np
import polars as pl
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import average_precision_score, precision_recall_fscore_support

from backend.core.config import get_settings
from backend.services.features.feature_contract import FeatureContract
from backend.services.features.online_feature_engine import OnlineFeatureEngine, group_by_timestamp
from backend.services.features.snapshot import load_snapshot
from backend.services.inference.model_loader import load_xgboost_model
from backend.services.inference.xgboost_scorer import XGBoostScorer
from backend.services.inference.risk_policy import RiskPolicy
from backend.core.schemas import TransactionEvent

def run_replay(args):
    settings = get_settings()
    base_dir = Path(__file__).resolve().parents[2]
    
    if args.split == "test" and not args.allow_test:
        print("WARNING: Test split execution requires --allow-test flag. Thresholds must be frozen before proceeding.")
        exit(1)
        
    split_file = base_dir / f"data/processed/hi_small/modeling/splits/{args.split}.parquet"
    out_dir = Path(args.out) if args.out else base_dir / "results/reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading split: {split_file}")
    df = pl.scan_parquet(split_file).sort(["timestamp", "transaction_id"]).collect(engine="streaming")
    
    if args.max_events:
        df = df.head(args.max_events)
        
    labels_dict = dict(zip(df.get_column("transaction_id").to_list(), df.get_column("is_laundering").to_list()))
    
    contract = FeatureContract()
    engine = OnlineFeatureEngine()
    if args.warm_start:
        print(f"Loading snapshot from {args.warm_start}...")
        engine = load_snapshot(Path(args.warm_start))
        
    model = load_xgboost_model(contract=contract)
    scorer = XGBoostScorer(model, contract)
    
    assert settings.risk_policy is not None, "Risk policy config is missing"
    policy = RiskPolicy(settings.risk_policy)
    
    print("Preparing Event Stream...")
    raw_events = [
        TransactionEvent(event_id=f"evt_{row['transaction_id']}", **row) 
        for row in df.with_columns(pl.col("timestamp").dt.to_string("%Y-%m-%dT%H:%M:%SZ")).to_dicts()
    ]
    batches = group_by_timestamp(raw_events)
    
    results = []
    latencies = []
    feature_memory = {} # Used to debug if drift occurs
    
    print("Starting chronological replay...")
    for batch in tqdm(batches, desc="Replaying"):
        t0 = time.perf_counter()
        features = engine.process_batch(batch)
        f_latency = (time.perf_counter() - t0) * 1000.0
        
        if not features:
            continue
            
        s0 = time.perf_counter()
        scores = scorer.score_rows(features)
        s_latency = (time.perf_counter() - s0) * 1000.0
        
        p0 = time.perf_counter()
        for idx, row in enumerate(features):
            feature_memory[row.transaction_id] = row.features
            decision = policy.process_decision(
                transaction_id=row.transaction_id,
                risk_score=float(scores[idx]),
                model_version=scorer.model_version,
                feature_version="f4_v1",
                latency_ms=(f_latency + s_latency + (time.perf_counter() - p0) * 1000.0) / len(features)
            )
            results.append(decision)
            latencies.append(decision.latency_ms)

    y_true = np.array([labels_dict[r.transaction_id] for r in results])
    y_scores = np.array([r.risk_score for r in results])
    
    has_positives = y_true.sum() > 0
    pr_auc = average_precision_score(y_true, y_scores) if has_positives else 0.0
    
    alerts = [r for r in results if r.decision == "alert"]
    escalations = [r for r in results if r.escalation_required]
    deferred = [r for r in results if r.investigation_deferred]
    
    if has_positives:
        pred_labels = (y_scores >= settings.risk_policy.alert_threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, pred_labels, average='binary', zero_division=0)
    else:
        precision, recall, f1 = 0.0, 0.0, 0.0
    
    k = min(100, len(y_scores))
    top_k_indices = np.argsort(y_scores)[::-1][:k]
    precision_at_100 = y_true[top_k_indices].mean() if k > 0 else 0.0
    
    summary = {
        "split": args.split,
        "events_processed": len(results),
        "mean_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
        "p95_latency_ms": float(np.percentile(latencies, 95)) if latencies else 0.0,
        "alerts_generated": len(alerts),
        "escalations_triggered": len(escalations),
        "investigations_deferred": len(deferred),
        "metrics": {
            "pr_auc": float(pr_auc),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "precision_at_100": float(precision_at_100)
        }
    }
    
    with open(out_dir / "replay_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
        
    pl.DataFrame(
        [r.model_dump() for r in results],
        schema_overrides={"escalation_reason": pl.String}
    ).write_parquet(out_dir / "replay_decisions.parquet")
    
    print("\n--- Summary ---")
    print(json.dumps(summary, indent=2))
    
    if args.split == "validation":
        val_pred_file = base_dir / "results/models/xgboost_f4_v1/validation_predictions.parquet"
        if val_pred_file.exists():
            print("\nComparing against offline baseline predictions...")
            offline = pl.scan_parquet(val_pred_file).collect()
            offline_scores = dict(zip(offline.get_column("transaction_id").to_list(), offline.get_column("xgboost_probability").to_list()))
            
            max_diff = 0.0
            worst_tx = None
            for r in results:
                if r.transaction_id in offline_scores:
                    diff = abs(r.risk_score - offline_scores[r.transaction_id])
                    if diff > max_diff:
                        max_diff = diff
                        worst_tx = r.transaction_id
            
            print(f"Max absolute probability difference vs offline: {max_diff:.8f}")
            if max_diff > 1e-5 and worst_tx is not None:
                print(f"WARNING: Replay predictions drift beyond tolerance! Diagnosing worst transaction: {worst_tx}")
                # Auto-diagnose feature mismatch
                off_row = df.filter(pl.col("transaction_id") == worst_tx).to_dicts()[0]
                on_row = feature_memory[worst_tx]
                print(f"\n--- Diagnostic Feature Mismatch (TX {worst_tx}) ---")
                for col in contract.numeric_features:
                    val_online = on_row.get(col)
                    val_offline = off_row.get(col)
                    
                    if (val_online is None or math.isnan(val_online)) and (val_offline is None or math.isnan(val_offline)):
                        continue
                    
                    if val_online is None or val_offline is None or math.isnan(val_online) or math.isnan(val_offline):
                        print(f"[!] {col}: Online={val_online}, Offline={val_offline}")
                        continue
                        
                    if not math.isclose(val_online, val_offline, rel_tol=1e-5, abs_tol=1e-5):
                        print(f"[!] {col}: Online={val_online}, Offline={val_offline}")
        else:
            print("Offline baseline predictions not found for comparison.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="validation", choices=["train", "validation", "test", "tail_stress"])
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--warm-start", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--allow-test", action="store_true")
    run_replay(parser.parse_args())