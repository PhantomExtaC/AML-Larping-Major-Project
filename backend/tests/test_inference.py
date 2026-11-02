import pytest
from backend.core.config import RiskPolicyConfig, BorderlineBand, RiskBands
from backend.services.inference.risk_policy import RiskPolicy, EscalationReason

@pytest.fixture
def risk_config():
    return RiskPolicyConfig(
        alert_threshold=0.95,
        escalation_threshold=0.03,
        borderline_band=BorderlineBand(low=0.01, high=0.03),
        investigation_budget_per_10k=5000, 
        max_queue_depth=10,
        risk_bands=RiskBands(low=0.01, medium=0.03, high=0.50, critical=0.95)
    )

def test_policy_bands_and_decisions(risk_config):
    policy = RiskPolicy(risk_config)
    
    d1 = policy.process_decision(1, 0.005, "v1", "f1", 5.0)
    assert d1.risk_band == "low"
    assert d1.decision == "clear"
    assert not d1.escalation_required
    
    d2 = policy.process_decision(2, 0.02, "v1", "f1", 5.0)
    assert d2.risk_band == "low"
    assert d2.decision == "monitor"
    assert d2.escalation_required
    assert d2.escalation_reason == EscalationReason.BORDERLINE.value
    
    d3 = policy.process_decision(3, 0.99, "v1", "f1", 5.0)
    assert d3.risk_band == "critical"
    assert d3.decision == "alert"
    assert d3.escalation_required
    assert d3.escalation_reason == EscalationReason.HIGH_SCORE.value

def test_investigation_budget_limits(risk_config):
    risk_config.investigation_budget_per_10k = 0
    risk_config.max_queue_depth = 2
    policy = RiskPolicy(risk_config)
    
    d1 = policy.process_decision(1, 0.99, "v1", "f1", 5.0)
    assert d1.escalation_required and not d1.investigation_deferred
    
    d2 = policy.process_decision(2, 0.99, "v1", "f1", 5.0)
    assert d2.escalation_required and not d2.investigation_deferred
    
    d3 = policy.process_decision(3, 0.99, "v1", "f1", 5.0)
    assert d3.escalation_required and d3.investigation_deferred