from enum import Enum
from typing import Any, Dict, List, Literal, Union, Optional
from pydantic import BaseModel

class TransactionEvent(BaseModel):
    event_id: str
    transaction_id: int
    timestamp: str
    from_bank: str
    from_account: str
    sender_id: str
    to_bank: str
    to_account: str
    receiver_id: str
    amount_paid: float
    amount_paid_log: float
    payment_currency: str
    amount_received: float
    amount_received_log: float
    receiving_currency: str
    same_currency_amount_delta: Optional[float] = None
    payment_format: str
    is_cross_bank: int
    is_currency_conversion: int
    is_self_transfer: int
    hour: int
    day_of_week: int
    day_of_month: int
    sim_metadata: Optional[Dict[str, Any]] = None

class GroundTruthRecord(BaseModel):
    transaction_id: int
    is_laundering: int
    scenario_id: Optional[str] = None
    scenario_type: Optional[str] = None
    scenario_version: Optional[str] = None
    role: Optional[str] = None
    difficulty: Optional[str] = None

class FeatureRow(BaseModel):
    transaction_id: int
    features: Dict[str, Any]

class DecisionRecord(BaseModel):
    transaction_id: int
    risk_score: float
    risk_band: str
    decision: str
    escalation_required: bool
    escalation_reason: Optional[str] = None
    investigation_deferred: bool = False
    model_version: str
    feature_version: str
    threshold_version: str
    scoring_timestamp: str
    latency_ms: float

class GraphNode(BaseModel):
    account_id: str
    node_features: Dict[str, float]

class GraphEdge(BaseModel):
    src: str
    dst: str
    timestamp: str
    edge_features: Dict[str, float]

class Subgraph(BaseModel):
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    center_accounts: List[str]
    as_of_timestamp: str
    build_params: Dict[str, Any]
    coverage_stats: Dict[str, Any]

class GraphAvailability(str, Enum):
    AVAILABLE = "graph_available"
    PARTIALLY_AVAILABLE = "graph_partially_available"
    UNAVAILABLE_COLD_START = "graph_unavailable_cold_start"

class GraphScore(BaseModel):
    score: Optional[float]
    availability: GraphAvailability
    model_version: str
    details: Dict[str, Any]

class InvestigationResult(BaseModel):
    investigation_id: str
    transaction_id: int
    graph_risk_score: Optional[float]
    graph_availability: GraphAvailability
    investigation_status: str
    nodes_examined: int
    edges_examined: int
    lookback_window: int
    max_hops: int
    graph_statistics: Dict[str, Any]
    suspicious_patterns: List[str]
    relevant_counterparties: List[str]
    model_version: str
    processing_latency: float
    error: Optional[str] = None

class FeatureContribution(BaseModel):
    feature_name: str
    feature_value: Union[float, str]
    shap_value: float

class ShapExplanation(BaseModel):
    base_value: float
    margin: float
    top_positive: List[FeatureContribution]
    top_negative: List[FeatureContribution]
    grouped_categorical: Dict[str, float]
    metadata: Dict[str, Any]

class InvestigationReport(BaseModel):
    transaction: TransactionEvent
    decision: DecisionRecord
    investigation: InvestigationResult
    explanation: ShapExplanation

class SimulationStatus(BaseModel):
    active: bool
    processed_count: int
    current_time: str
    alerts_generated: int

class BaseStreamMessage(BaseModel):
    timestamp: str

class TransactionScoredMessage(BaseStreamMessage):
    message_type: Literal["transaction_scored"] = "transaction_scored"
    transaction_id: int
    risk_score: float
    decision: str

class AlertCreatedMessage(BaseStreamMessage):
    message_type: Literal["alert_created"] = "alert_created"
    transaction_id: int
    risk_band: str

class InvestigationUpdatedMessage(BaseStreamMessage):
    message_type: Literal["investigation_updated"] = "investigation_updated"
    investigation_id: str
    status: str

class SimulationStatusMessage(BaseStreamMessage):
    message_type: Literal["simulation_status"] = "simulation_status"
    status: SimulationStatus

class ErrorMessage(BaseStreamMessage):
    message_type: Literal["error"] = "error"
    error_code: str
    details: str

StreamMessage = Union[
    TransactionScoredMessage,
    AlertCreatedMessage,
    InvestigationUpdatedMessage,
    SimulationStatusMessage,
    ErrorMessage
]