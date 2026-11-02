import hashlib
import json
import datetime
from enum import Enum
from typing import Tuple

from backend.core.config import RiskPolicyConfig
from backend.core.schemas import DecisionRecord

class EscalationReason(str, Enum):
    HIGH_SCORE = "HIGH_SCORE"
    BORDERLINE = "BORDERLINE"
    GRAPH_RULE = "GRAPH_RULE"
    NONE = "NONE"

class RiskPolicy:
    def __init__(self, config: RiskPolicyConfig):
        self.config = config
        
        config_str = json.dumps(config.model_dump(), sort_keys=True)
        self.threshold_version = "pol_" + hashlib.md5(config_str.encode('utf-8')).hexdigest()[:8]
        
        self.max_queue_depth = config.max_queue_depth
        self.budget_rate = config.investigation_budget_per_10k / 10000.0
        self.available_tokens = float(self.max_queue_depth)

    def _determine_band(self, score: float) -> str:
        bands = self.config.risk_bands
        if score >= bands.critical:
            return "critical"
        elif score >= bands.high:
            return "high"
        elif score >= bands.medium:
            return "medium"
        return "low"

    def _determine_escalation(self, score: float) -> Tuple[bool, EscalationReason]:
        if score >= self.config.escalation_threshold:
            return True, EscalationReason.HIGH_SCORE
            
        if self.config.borderline_band.low <= score <= self.config.borderline_band.high:
            return True, EscalationReason.BORDERLINE
            
        return False, EscalationReason.NONE

    def process_decision(
        self, 
        transaction_id: int, 
        risk_score: float, 
        model_version: str, 
        feature_version: str, 
        latency_ms: float,
        graph_rule_triggered: bool = False
    ) -> DecisionRecord:
        
        self.available_tokens = min(float(self.max_queue_depth), self.available_tokens + self.budget_rate)
        
        risk_band = self._determine_band(risk_score)
        decision = "alert" if risk_score >= self.config.alert_threshold else "monitor" if risk_score >= self.config.borderline_band.low else "clear"
        
        escalation_required, escalation_reason = self._determine_escalation(risk_score)
        
        if not escalation_required and graph_rule_triggered:
            escalation_required = True
            escalation_reason = EscalationReason.GRAPH_RULE
            
        investigation_deferred = False
        if escalation_required:
            if self.available_tokens >= 1.0:
                self.available_tokens -= 1.0
            else:
                investigation_deferred = True
                
        now_utc = datetime.datetime.now(datetime.timezone.utc)
                
        return DecisionRecord(
            transaction_id=transaction_id,
            risk_score=risk_score,
            risk_band=risk_band,
            decision=decision,
            escalation_required=escalation_required,
            escalation_reason=escalation_reason.value if escalation_required else None,
            investigation_deferred=investigation_deferred,
            model_version=model_version,
            feature_version=feature_version,
            threshold_version=self.threshold_version,
            scoring_timestamp=now_utc.isoformat().replace("+00:00", "Z"),
            latency_ms=latency_ms
        )