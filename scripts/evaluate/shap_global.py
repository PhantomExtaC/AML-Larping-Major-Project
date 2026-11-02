import json
import argparse
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr

from backend.core.config import get_settings
from backend.core.schemas import FeatureRow
from backend.services.inference.model_loader import load_xgboost_model
from backend.services.inference.xgboost_scorer import XGBoostScorer
from backend.services.explainability.global_importance import compute_global_importance

def run_global_shap(args):
    settings = get_settings()
    base_dir = Path(__file__).resolve().parents[2]
    
    val_file = base_dir / "data/processed/hi_small/modeling/splits/validation.parquet"
    out_dir = base_dir / "results/models/xgboost_f4_v1"
    plot_dir = base_dir / "results/plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading Validation Split...")
    val_df = pl.scan_parquet(val_file).collect()
    
    pos_df = val_df.filter(pl.col("is_laundering") == 1)
    neg_df = val_df.filter(pl.col("is_laundering") == 0)
    
    n_pos = len(pos_df)
    n_neg_sample = args.samples - n_pos
    
    true_prevalence = n_pos / len(val_df)
    print(f"Validation True Prevalence: {true_prevalence:.6f}")
    
    sampled_df = pl.concat([
        pos_df,
        neg_df.sample(n=n_neg_sample, seed=42)
    ])
    
    print(f"Stratified sample: {len(sampled_df)} rows. Computing global SHAP...")
    
    model = load_xgboost_model()
    scorer = XGBoostScorer(model, model.contract)
    
    # Correctly route through the FeatureContract to generate one-hot columns
    raw_dicts = sampled_df.to_dicts()
    feature_rows = [
        FeatureRow(transaction_id=row["transaction_id"], features=row) 
        for row in raw_dicts
    ]
    
    matrix = scorer.contract.to_matrix(feature_rows)
    y_true = sampled_df.get_column("is_laundering").to_numpy()
    
    importance_data = compute_global_importance(scorer, matrix, y_true, true_prevalence)
    
    gain_scores = scorer.model.get_score(importance_type='gain')
    
    ungrouped = importance_data["ungrouped_importance"]
    grouped = importance_data["grouped_importance"]
    
    shap_vals, gain_vals = [], []
    for f in model.contract.feature_names:
        shap_vals.append(ungrouped[f])
        gain_vals.append(gain_scores.get(f, 0.0))
        
    corr, _ = spearmanr(shap_vals, gain_vals)
    importance_data["rank_correlation_with_gain"] = float(corr)
    
    print(f"Spearman Rank Correlation (|SHAP| vs Gain): {corr:.4f}")
    
    out_file = out_dir / "shap_global.json"
    with open(out_file, "w") as f:
        json.dump(importance_data, f, indent=2)
    print(f"Saved {out_file}")
    
    top_20 = sorted(grouped.items(), key=lambda x: x[1], reverse=True)[:20]
    names = [x[0] for x in top_20][::-1]
    vals = [x[1] for x in top_20][::-1]
    
    plt.figure(figsize=(12, 8))
    plt.barh(names, vals, color='#17a2b8')
    plt.title("Top 20 Features by Mean |SHAP| (Grouped, Prevalence-Reweighted)")
    plt.xlabel("Mean |SHAP| (Log-Odds Space)")
    plt.tight_layout()
    
    plot_file = plot_dir / "shap_global_importance.png"
    plt.savefig(plot_file, dpi=150)
    print(f"Saved {plot_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=5000, help="Total samples (includes all positives)")
    args = parser.parse_args()
    run_global_shap(args)