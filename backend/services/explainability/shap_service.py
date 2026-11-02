import math
import yaml
import numpy as np
import xgboost as xgb
from pathlib import Path
from typing import List, Dict

from backend.core.schemas import FeatureRow, ShapExplanation, FeatureContribution
from backend.services.inference.xgboost_scorer import XGBoostScorer

class ShapService:
    DISCLAIMER = (
        "Contributions describe the model's behaviour in log-odds space; they are associations "
        "in a synthetic dataset, not proof or causal explanations. Scores are uncalibrated."
    )

    def __init__(self, scorer: XGBoostScorer):
        self.scorer = scorer
        self.contract = scorer.contract
        self.feature_names = self.contract.feature_names
        self.labels = self._load_labels()
        
        self.categorical_groups: Dict[str, List[int]] = {}
        col_idx_map = {name: idx for idx, name in enumerate(self.feature_names)}
        
        for source_col, cat_val, out_col in self.contract.one_hot_features:
            if source_col not in self.categorical_groups:
                self.categorical_groups[source_col] = []
            if out_col in col_idx_map:
                self.categorical_groups[source_col].append(col_idx_map[out_col])

    def _load_labels(self) -> Dict[str, str]:
        base_dir = Path(__file__).resolve().parents[3]
        label_file = base_dir / "backend/configs/feature_labels.yaml"
        labels = {}
        if label_file.exists():
            with open(label_file, "r") as f:
                data = yaml.safe_load(f)
                if data and "overrides" in data:
                    labels = data["overrides"]
        return labels

    def _get_display_name(self, feature_name: str) -> str:
        if feature_name in self.labels:
            return self.labels[feature_name]
        return feature_name.replace("_", " ").title()

    def explain(self, row: FeatureRow, top_k: int = 10) -> ShapExplanation:
        return self.explain_batch([row], top_k=top_k)[0]

    def explain_batch(self, rows: List[FeatureRow], top_k: int = 10) -> List[ShapExplanation]:
        if not rows:
            return []
            
        matrix = self.contract.to_matrix(rows)
        dmatrix = xgb.DMatrix(matrix, feature_names=self.feature_names, missing=np.nan)
        
        contribs = self.scorer.model.predict(dmatrix, pred_contribs=True)
        scores = self.scorer.model.predict(dmatrix)
        
        explanations = []
        for i, row in enumerate(rows):
            shap_vals = contribs[i, :-1]
            base_value = float(contribs[i, -1])
            margin = float(np.sum(shap_vals) + base_value)
            
            grouped_view = {}
            for cat_col, indices in self.categorical_groups.items():
                group_sum = float(np.sum(shap_vals[indices]))
                active_val = row.features.get(cat_col, "Unknown")
                grouped_view[f"{cat_col} ({active_val})"] = group_sum
                
            sorted_indices = np.argsort(shap_vals)
            top_negative_idx = sorted_indices[:top_k]
            top_positive_idx = sorted_indices[-top_k:][::-1]
            
            def build_contributions(indices: np.ndarray) -> List[FeatureContribution]:
                out = []
                for idx in indices:
                    val = float(matrix[i, idx])
                    f_name = self.feature_names[idx]
                    display_val = "missing" if math.isnan(val) else val
                        
                    out.append(FeatureContribution(
                        feature_name=self._get_display_name(f_name),
                        feature_value=display_val,
                        shap_value=float(shap_vals[idx])
                    ))
                return out

            explanations.append(ShapExplanation(
                base_value=base_value,
                margin=margin,
                top_positive=build_contributions(top_positive_idx),
                top_negative=build_contributions(top_negative_idx),
                grouped_categorical=grouped_view,
                metadata={
                    "model_version": self.scorer.model_version,
                    "feature_version": "f4_v1",
                    "score": float(scores[i]),
                    "disclaimer": self.DISCLAIMER
                }
            ))
            
        return explanations