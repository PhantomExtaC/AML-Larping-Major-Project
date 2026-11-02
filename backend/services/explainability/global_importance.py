import numpy as np
import xgboost as xgb
from typing import Dict, Any

from backend.services.inference.xgboost_scorer import XGBoostScorer

def compute_global_importance(
    scorer: XGBoostScorer, 
    matrix: np.ndarray, 
    y_true: np.ndarray, 
    true_prevalence: float
) -> Dict[str, Any]:
    
    dmatrix = xgb.DMatrix(matrix, feature_names=scorer.contract.feature_names, missing=np.nan)
    contribs = scorer.model.predict(dmatrix, pred_contribs=True)[:, :-1]
    
    mean_abs_shap = np.mean(np.abs(contribs), axis=0)
    
    pos_mask = (y_true == 1)
    neg_mask = (y_true == 0)
    sample_prevalence = pos_mask.mean()
    
    weights = np.ones(len(y_true), dtype=float)
    if 0 < sample_prevalence < 1:
        w_pos = true_prevalence / sample_prevalence
        w_neg = (1.0 - true_prevalence) / (1.0 - sample_prevalence)
        weights = np.where(pos_mask, w_pos, w_neg)
        weighted_abs_shap = np.average(np.abs(contribs), axis=0, weights=weights)
    else:
        weighted_abs_shap = mean_abs_shap

    ungrouped = {
        name: float(val) for name, val in zip(scorer.contract.feature_names, weighted_abs_shap)
    }
    
    grouped_contribs = {}
    used_indices = set()
    col_idx_map = {name: idx for idx, name in enumerate(scorer.contract.feature_names)}
    
    for source_col, _, out_col in scorer.contract.one_hot_features:
        if source_col not in grouped_contribs:
            grouped_contribs[source_col] = []
        if out_col in col_idx_map:
            idx = col_idx_map[out_col]
            grouped_contribs[source_col].append(idx)
            used_indices.add(idx)
        
    grouped_importance = {}
    
    for i, name in enumerate(scorer.contract.feature_names):
        if i not in used_indices:
            grouped_importance[name] = float(weighted_abs_shap[i])
            
    for group_name, indices in grouped_contribs.items():
        transaction_group_sums = np.sum(contribs[:, indices], axis=1)
        group_mean_abs = np.average(np.abs(transaction_group_sums), weights=weights)
        grouped_importance[group_name] = float(group_mean_abs)

    return {
        "ungrouped_importance": ungrouped,
        "grouped_importance": grouped_importance
    }