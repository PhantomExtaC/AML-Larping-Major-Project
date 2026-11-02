import argparse
import json
import math
from pathlib import Path
import polars as pl
import numpy as np

from backend.core.config import get_settings
from backend.services.features.online_feature_engine import OnlineFeatureEngine, group_by_timestamp
from backend.services.features.feature_contract import FeatureContract
from backend.core.schemas import TransactionEvent

# Validating account union replay:
# The subset of events touching a specific set of target accounts contains EVERY historical 
# operation necessary to build those specific accounts' features (and pair features). 
# Filtering the global event log down to this ego-graph preserves complete timeline correctness 
# without leaking features from unrelated accounts or dropping dependent metrics like unique counterparties.

def run_parity(args):
    settings = get_settings()
    base_dir = Path(__file__).resolve().parents[2]
    
    validation_file = base_dir / "data/processed/hi_small/modeling/splits/validation.parquet"
    raw_tx_pattern = str(base_dir / "data/processed/hi_small/transactions/*.parquet")
    phase2b_file = base_dir / "data/processed/hi_small/modeling/phase2b/features.parquet"
    output_report = base_dir / "results/reports/feature_parity_report.json"
    output_report.parent.mkdir(parents=True, exist_ok=True)
    
    print("Selecting target transactions from validation...")
    val_df = pl.scan_parquet(validation_file).collect()
    
    # Stratified sampling
    pos_df = val_df.filter(pl.col("is_laundering") == 1)
    neg_df = val_df.filter(pl.col("is_laundering") == 0)
    
    n_pos = min(len(pos_df), args.samples // 2)
    n_neg = args.samples - n_pos
    
    sampled_df = pl.concat([
        pos_df.sample(n=n_pos, seed=42),
        neg_df.sample(n=n_neg, seed=42)
    ])
    
    target_tx_ids = set(sampled_df.get_column("transaction_id").to_list())
    target_accounts = set(sampled_df.get_column("sender_id").to_list()) | set(sampled_df.get_column("receiver_id").to_list())
    
    print(f"Sampled {len(target_tx_ids)} transactions covering {len(target_accounts)} unique accounts.")
    
    print("Extracting full chronological ego-graph for target accounts...")
    ego_events_df = (
        pl.scan_parquet(raw_tx_pattern)
        .filter(
            pl.col("sender_id").is_in(list(target_accounts)) | 
            pl.col("receiver_id").is_in(list(target_accounts))
        )
        .with_columns([
            pl.col("timestamp").dt.to_string("%Y-%m-%dT%H:%M:%SZ").alias("timestamp"),
            pl.lit("offline-replay").alias("event_id")
        ])
        .sort(["timestamp", "transaction_id"])
        .collect(engine="streaming")
    )
    
    print(f"Extracted {len(ego_events_df)} historical events. Grouping by timestamp...")
    raw_events = [TransactionEvent(**row) for row in ego_events_df.to_dicts()]
    batches = group_by_timestamp(raw_events)
    
    engine = OnlineFeatureEngine()
    contract = FeatureContract()
    
    computed_rows = {}
    print("Replaying through OnlineFeatureEngine...")
    for batch in batches:
        features = engine.process_batch(batch)
        for row in features:
            if row.transaction_id in target_tx_ids:
                computed_rows[row.transaction_id] = row
                
    print("Loading Offline Phase 2B features for exact comparison...")
    offline_features_df = (
        pl.scan_parquet(phase2b_file)
        .filter(pl.col("transaction_id").is_in(list(target_tx_ids)))
        .collect(engine="streaming")
    )
    offline_features = {row["transaction_id"]: row for row in offline_features_df.to_dicts()}
    
    mismatches = 0
    detailed_errors = []
    
    # Compare raw base features (Phase 2B full contract)
    check_columns = [
        col for col in contract.numeric_features
    ] + ["sender_amount_mean_24h", "receiver_amount_variance_24h", "sender_amount_sumsq_24h"] # Spot check internal helpers
    check_columns = [c for c in check_columns if c in offline_features_df.columns]
    
    print("Evaluating Parity...")
    for tx_id in target_tx_ids:
        online_f = computed_rows[tx_id].features
        offline_f = offline_features[tx_id]
        
        for col in check_columns:
            val_online = online_f.get(col)
            val_offline = offline_f.get(col)
            
            # Both NaN
            if (val_online is None or math.isnan(val_online)) and (val_offline is None or math.isnan(val_offline)):
                continue
                
            if val_online is None or val_offline is None or math.isnan(val_online) or math.isnan(val_offline):
                mismatches += 1
                detailed_errors.append({"tx_id": tx_id, "col": col, "online": val_online, "offline": val_offline, "reason": "NaN mismatch"})
                continue
                
            if not math.isclose(val_online, val_offline, rel_tol=args.rel_tol, abs_tol=1e-5):
                mismatches += 1
                detailed_errors.append({"tx_id": tx_id, "col": col, "online": val_online, "offline": val_offline, "reason": "Value mismatch"})
                
    report = {
        "samples": len(target_tx_ids),
        "ego_graph_events": len(ego_events_df),
        "mismatches": mismatches,
        "first_mismatches": detailed_errors[:10]
    }
    
    with open(output_report, "w") as f:
        json.dump(report, f, indent=2)
        
    print(f"\nFeature Parity Report:")
    print(f"Tested Samples: {len(target_tx_ids)}")
    print(f"Mismatches: {mismatches}")
    if mismatches > 0:
        print("First 3 mismatches:")
        for err in detailed_errors[:3]:
            print(f"  TX {err['tx_id']} - {err['col']}: Online={err['online']}, Offline={err['offline']} ({err['reason']})")
        exit(1)
    else:
        print("Success! 100% exact parity achieved.")
        exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=5000, help="Number of validation transactions to sample")
    parser.add_argument("--rel_tol", type=float, default=1e-5, help="Relative tolerance for float comparisons")
    args = parser.parse_args()
    run_parity(args)