import argparse
from pathlib import Path
import polars as pl
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

def generate_report():
    base_dir = Path(__file__).resolve().parents[2]
    preds_file = base_dir / "results/models/xgboost_f4_v1/validation_predictions.parquet"
    out_dir = base_dir / "results/comparisons"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if not preds_file.exists():
        raise FileNotFoundError(f"Missing {preds_file}")
        
    df = pl.scan_parquet(preds_file).collect()
    y_true = df.get_column("true_label").to_numpy()
    y_scores = df.get_column("xgboost_probability").to_numpy()
    
    total_events = len(y_true)
    
    thresholds = {
        "Best F1": 0.95896,
        "Recall 80%": 0.03219,
        "Recall 95%": 0.001405
    }
    
    results = []
    
    for name, thresh in thresholds.items():
        preds = (y_scores >= thresh).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, preds, average='binary', zero_division=0)
        
        alerts = int(preds.sum())
        alerts_per_10k = (alerts / total_events) * 10000.0
        
        results.append({
            "Policy": name,
            "Threshold": thresh,
            "Precision": float(precision),
            "Recall": float(recall),
            "F1": float(f1),
            "Total Alerts": alerts,
            "Alerts per 10k": round(alerts_per_10k, 2)
        })
        
    res_df = pl.DataFrame(results)
    res_df.write_csv(out_dir / "policy_report.csv")
    
    markdown = "### Risk Policy Operational Report\n\n"
    markdown += "| Policy | Threshold | Precision | Recall | F1 | Total Alerts | Alerts/10k |\n"
    markdown += "|---|---|---|---|---|---|---|\n"
    for r in results:
        markdown += f"| {r['Policy']} | {r['Threshold']:.5f} | {r['Precision']:.4f} | {r['Recall']:.4f} | {r['F1']:.4f} | {r['Total Alerts']} | {r['Alerts per 10k']} |\n"
        
    with open(out_dir / "policy_report.md", "w") as f:
        f.write(markdown)
        
    print("Generated results/comparisons/policy_report.md")
    print(markdown)

if __name__ == "__main__":
    generate_report()