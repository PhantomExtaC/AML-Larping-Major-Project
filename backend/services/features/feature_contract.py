import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np

from backend.core.config import get_settings
from backend.core.schemas import FeatureRow

class FeatureContract:
    """
    Single source of truth for translating a dictionary of raw features 
    into a strictly ordered float32 NumPy matrix suitable for XGBoost.
    """
    
    def __init__(self, definition_path: Path | None = None):
        if definition_path is None:
            settings = get_settings()
            base_dir = Path(__file__).resolve().parents[3]
            definition_path = base_dir / settings.paths.feature_contract
            
        with open(definition_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        self.numeric_features: List[str] = data["numeric_features"]
        self.category_values: Dict[str, List[str]] = data["category_values"]
        # list of (source_col, category_value, final_feature_name)
        self.one_hot_features: List[Tuple[str, str, str]] = [tuple(x) for x in data["one_hot_features"]]
        self.feature_names: List[str] = data["feature_names"]
        
        # Build quick lookups for one-hot encoding
        self._one_hot_lookups: Dict[str, Dict[str, str]] = {col: {} for col in self.category_values.keys()}
        for source_col, cat_val, out_col in self.one_hot_features:
            self._one_hot_lookups[source_col][cat_val] = out_col

    def to_matrix(self, rows: List[FeatureRow]) -> np.ndarray:
        """
        Convert feature rows to an ordered float32 matrix.
        Unknown categories result in all-zero vectors.
        +/- inf values are mapped to NaN, matching offline logic.
        """
        if not rows:
            return np.empty((0, len(self.feature_names)), dtype=np.float32)
            
        num_rows = len(rows)
        num_cols = len(self.feature_names)
        matrix = np.zeros((num_rows, num_cols), dtype=np.float32)
        
        # Mapping from output feature name to matrix index for fast assignment
        col_idx_map = {name: idx for idx, name in enumerate(self.feature_names)}
        
        for i, row in enumerate(rows):
            features = row.features
            
            # 1. Numeric features
            for num_col in self.numeric_features:
                val = features.get(num_col)
                if val is None or math.isnan(val):
                    matrix[i, col_idx_map[num_col]] = np.nan
                elif math.isinf(val):
                    matrix[i, col_idx_map[num_col]] = np.nan
                else:
                    matrix[i, col_idx_map[num_col]] = val
                    
            # 2. Categorical features (One-hot)
            for cat_col, vocab in self._one_hot_lookups.items():
                cat_val = features.get(cat_col)
                # Ensure it's treated as string to match vocabulary
                if cat_val is not None:
                    cat_val_str = str(cat_val)
                    if cat_val_str in vocab:
                        out_col = vocab[cat_val_str]
                        matrix[i, col_idx_map[out_col]] = 1.0
                        
        return matrix