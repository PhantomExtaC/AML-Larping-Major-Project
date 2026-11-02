import yaml
from pathlib import Path
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, BaseModel

class PathsConfig(BaseModel):
    raw_data: str = "data/raw/ibm_aml/hi_small"
    processed_data: str = "data/processed/hi_small"
    xgboost_model: str = "research/models/baseline/xgboost_f4_v1/model.json"
    feature_contract: str = "research/models/baseline/xgboost_f4_v1/feature_definition.json"
    graph_exports: str = "data/processed/hi_small/graph_v1"

class ApiConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = ["http://localhost:5173"]

class StorageConfig(BaseModel):
    type: str = "in_memory"

class SimulationConfig(BaseModel):
    default_seed: int = 42

class RiskBands(BaseModel):
    low: float
    medium: float
    high: float
    critical: float

class BorderlineBand(BaseModel):
    low: float
    high: float

class RiskPolicyConfig(BaseModel):
    alert_threshold: float
    escalation_threshold: float
    borderline_band: BorderlineBand
    investigation_budget_per_10k: int
    max_queue_depth: int
    risk_bands: RiskBands

class InvestigationConfig(BaseModel):
    max_hops: int = 2
    lookback_window_days: int = 30
    timeout_ms: int = 5000
    enable_graph_analytics: bool = True

class GraphModelConfig(BaseModel):
    model_version: str = "graphsage_v1_mock"
    use_gpu: bool = False
    batch_size: int = 128
    embedding_dim: int = 128

class AppSettings(BaseSettings):
    paths: PathsConfig = Field(default_factory=PathsConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    risk_policy: RiskPolicyConfig | None = None
    investigation: InvestigationConfig | None = None
    graph_model: GraphModelConfig | None = None
    
    model_config = SettingsConfigDict(env_file=".env", env_nested_delimiter="__")

def _load_yaml(file_path: Path) -> dict:
    if not file_path.exists():
        return {}
    with open(file_path, "r") as f:
        return yaml.safe_load(f) or {}

@lru_cache
def get_settings(strict: bool = False) -> AppSettings:
    """
    Load settings from YAML files and .env overrides.
    If strict=True, validates that specified artifact paths exist.
    """
    base_dir = Path(__file__).resolve().parents[2]
    config_dir = base_dir / "backend" / "configs"
    
    app_data = _load_yaml(config_dir / "app.yaml")
    settings = AppSettings(**app_data)
    
    settings.risk_policy = RiskPolicyConfig(**_load_yaml(config_dir / "risk_policy.yaml"))
    settings.investigation = InvestigationConfig(**_load_yaml(config_dir / "investigation.yaml"))
    settings.graph_model = GraphModelConfig(**_load_yaml(config_dir / "graph_model.yaml"))
    
    if strict:
        for path_attr in ["xgboost_model", "feature_contract"]:
            path = base_dir / getattr(settings.paths, path_attr)
            if not path.exists():
                raise FileNotFoundError(f"Strict mode: Required artifact missing: {path}")
                
    return settings