import uuid
from backend.core.config import get_settings
from backend.core.ids import make_account_id, derive_rng
from backend.core.schemas import StreamMessage, TransactionScoredMessage, AlertCreatedMessage
from pydantic import TypeAdapter

def test_config_loads_defaults():
    settings = get_settings(strict=False)
    assert settings.api.port == 8000
    assert settings.risk_policy is not None
    assert settings.risk_policy.alert_threshold == 0.95896
    assert settings.investigation is not None
    assert settings.investigation.max_hops == 2

def test_make_account_id():
    assert make_account_id("BankA", "12345") == "BankA-12345"

def test_derive_rng_determinism():
    rng1 = derive_rng(42, "scenario_a", "agent_1")
    val1 = rng1.random()
    
    rng2 = derive_rng(42, "scenario_a", "agent_1")
    val2 = rng2.random()
    
    rng3 = derive_rng(42, "scenario_b", "agent_1")
    val3 = rng3.random()
    
    assert val1 == val2
    assert val1 != val3

def test_stream_message_discriminator():
    adapter = TypeAdapter(StreamMessage)
    
    msg1_json = '{"message_type": "transaction_scored", "timestamp": "2023-01-01T12:00:00Z", "transaction_id": 100, "risk_score": 0.85, "decision": "escalate"}'
    parsed1 = adapter.validate_json(msg1_json)
    assert isinstance(parsed1, TransactionScoredMessage)
    assert parsed1.transaction_id == 100
    
    msg2_json = '{"message_type": "alert_created", "timestamp": "2023-01-01T12:05:00Z", "transaction_id": 100, "risk_band": "high"}'
    parsed2 = adapter.validate_json(msg2_json)
    assert isinstance(parsed2, AlertCreatedMessage)
    assert parsed2.risk_band == "high"