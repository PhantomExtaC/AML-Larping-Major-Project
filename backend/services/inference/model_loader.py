import hashlib
from pathlib import Path
import xgboost as xgb

from backend.core.config import get_settings
from backend.services.features.feature_contract import FeatureContract

class LoadedModel:
    def __init__(self, booster: xgb.Booster, version_name: str, file_hash: str, xgb_version: str, contract: FeatureContract ):
        self.booster = booster
        self.version_name = version_name
        self.file_hash = file_hash
        self.xgb_version = xgb_version
        self.contract = contract

def load_xgboost_model(model_path: Path | None = None, contract: FeatureContract | None = None) -> LoadedModel:
    if model_path is None:
        settings = get_settings()
        base_dir = Path(__file__).resolve().parents[3]
        model_path = base_dir / settings.paths.xgboost_model
        
    if contract is None:
        contract = FeatureContract()
        
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
        
    hasher = hashlib.sha256()
    with open(model_path, "rb") as f:
        hasher.update(f.read())
    file_hash = hasher.hexdigest()[:12]
    
    booster = xgb.Booster()
    booster.load_model(model_path)
    
    expected_features = booster.feature_names
    if expected_features is not None:
        if len(expected_features) != len(contract.feature_names):
            raise ValueError(
                f"Feature count mismatch: model expects {len(expected_features)}, "
                f"contract defines {len(contract.feature_names)}"
            )
        if expected_features != contract.feature_names:
            raise ValueError("Feature ordering or names do not match strictly between model and contract.")
            
    if booster.num_features() != 132:
         raise ValueError(f"Model expects {booster.num_features()} features, but contract mandates strictly 132.")
    
    version_name = model_path.parent.name

    return LoadedModel(
        booster=booster,
        version_name=version_name,
        file_hash=file_hash,
        xgb_version=xgb.__version__,
        contract=contract
    )