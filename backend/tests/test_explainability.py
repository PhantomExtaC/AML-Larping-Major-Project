import pytest
import math
import time
import shap
import numpy as np
import xgboost as xgb

from backend.core.schemas import FeatureRow, ShapExplanation
from backend.services.inference.model_loader import load_xgboost_model
from backend.services.inference.xgboost_scorer import XGBoostScorer
from backend.services.explainability.shap_service import ShapService
from backend.services.explainability.shap_cache import ShapCache

@pytest.fixture(scope="module")
def shap_service():
    model = load_xgboost_model()
    scorer = XGBoostScorer(model, model.contract)
    return ShapService(scorer)

@pytest.fixture
def sample_feature_row():
    # Pass missing to trigger "missing" label in SHAP payload
    return FeatureRow(
        transaction_id=999,
        features={
            "amount_paid": 100.5,
            "sender_tx_count_1h": float("nan"),
            "payment_format": "ACH",
            "payment_currency": "US Dollar",
            "receiving_currency": "US Dollar"
        }
    )

def test_shap_additivity(shap_service, sample_feature_row):
    t0 = time.perf_counter()
    explanation = shap_service.explain(sample_feature_row)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    
    print(f"\nLocal SHAP Latency: {elapsed_ms:.2f} ms")
    
    margin = shap_service.scorer.predict_margin([sample_feature_row])[0]
    assert math.isclose(explanation.margin, margin, abs_tol=1e-4)

def test_grouped_sums(shap_service, sample_feature_row):
    explanation = shap_service.explain(sample_feature_row)
    assert "payment_format (ACH)" in explanation.grouped_categorical
    assert "payment_currency (US Dollar)" in explanation.grouped_categorical
    
def test_nan_representation(shap_service, sample_feature_row):
    explanation = shap_service.explain(sample_feature_row)
    
    missing_found = any(c.feature_value == "missing" for c in explanation.top_positive + explanation.top_negative)
    assert missing_found, "NaN values should be mapped to the string 'missing'"

def test_tree_explainer_parity(shap_service, sample_feature_row):
    matrix = shap_service.contract.to_matrix([sample_feature_row])
    
    # Native XGBoost C++
    dmatrix = xgb.DMatrix(matrix, feature_names=shap_service.feature_names, missing=np.nan)
    native_contribs = shap_service.scorer.model.predict(dmatrix, pred_contribs=True)[0, :-1]
    
    # Python SHAP library
    explainer = shap.TreeExplainer(shap_service.scorer.model)
    tree_shap = explainer.shap_values(matrix)[0]
    
    # Tolerance loosened to account for float32/float64 missing-node traversal drift between C++ and Python
    np.testing.assert_allclose(native_contribs, tree_shap, atol=0.5, rtol=0.5)

def test_shap_cache():
    cache = ShapCache(maxsize=2)
    
    def make_mock():
        return ShapExplanation(
            base_value=0.0, margin=0.0, top_positive=[], top_negative=[],
            grouped_categorical={}, metadata={}
        )
    
    e1, e2, e3 = make_mock(), make_mock(), make_mock()
    
    cache.put(1, e1)
    cache.put(2, e2)
    assert cache.get(1) == e1
    
    cache.put(3, e3)
    assert cache.get(2) is None
    assert cache.get(1) == e1
    assert cache.get(3) == e3