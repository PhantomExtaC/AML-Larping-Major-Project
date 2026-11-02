import time
import numpy as np
import xgboost as xgb
from typing import List

from backend.core.schemas import FeatureRow
from backend.services.inference.model_loader import LoadedModel
from backend.services.features.feature_contract import FeatureContract

class XGBoostScorer:
    def __init__(self, loaded_model: LoadedModel, contract: FeatureContract):
        self.model = loaded_model.booster
        self.model_version = f"{loaded_model.version_name}-{loaded_model.file_hash}"
        self.contract = contract
        
        self.total_scored = 0
        self.total_latency_ms = 0.0

    def score_rows(self, feature_rows: List[FeatureRow]) -> np.ndarray:
        if not feature_rows:
            return np.array([], dtype=np.float32)
            
        t0 = time.perf_counter()
        
        matrix = self.contract.to_matrix(feature_rows)
        # Explicitly declare missing=np.nan to ensure dense numpy NaNs match offline Polars nulls
        dmatrix = xgb.DMatrix(matrix, feature_names=self.contract.feature_names, missing=np.nan)
        
        predictions = self.model.predict(dmatrix)
        
        latency_ms = (time.perf_counter() - t0) * 1000.0
        self.total_scored += len(feature_rows)
        self.total_latency_ms += latency_ms
        
        return predictions

    def predict_margin(self, feature_rows: List[FeatureRow]) -> np.ndarray:
        if not feature_rows:
            return np.array([], dtype=np.float32)
            
        matrix = self.contract.to_matrix(feature_rows)
        dmatrix = xgb.DMatrix(matrix, feature_names=self.contract.feature_names, missing=np.nan)
        
        return self.model.predict(dmatrix, output_margin=True)
        
    def get_average_latency_ms(self) -> float:
        if self.total_scored == 0:
            return 0.0
        return self.total_latency_ms / self.total_scored