import pytest
import numpy as np
import math
from backend.services.features.rolling_windows import RollingStatWindow
from backend.services.features.online_feature_engine import OnlineFeatureEngine
from backend.core.schemas import TransactionEvent

def test_rolling_window_boundaries():
    rw = RollingStatWindow(max_window=3600.0) # 1 hour
    
    # Event exactly at t-1h IS INCLUDED in Polars closed='left' window [t-1h, t).
    # current_ts = 7200, window = 3600. 7200 - 3600 <= 3600 (True).
    rw.add(3600.0, 100.0) 
    rw.add(5000.0, 50.0)
    
    cnt, total, _ = rw.get_stats(7200.0, window=3600.0)
    assert cnt == 2
    assert total == 150.0
    
    # Move forward, 3600 is pruned out
    cnt, total, _ = rw.get_stats(7201.0, window=3600.0)
    assert cnt == 1
    assert total == 50.0

def test_engine_batch_isolation():
    engine = OnlineFeatureEngine()
    
    e1 = TransactionEvent(
        event_id="e1", transaction_id=1, timestamp="2023-01-01T10:00:00Z",
        from_bank="A", from_account="1", sender_id="A-1", to_bank="B", to_account="1", receiver_id="B-1",
        amount_paid=100.0, amount_paid_log=4.6, payment_currency="US Dollar",
        amount_received=100.0, amount_received_log=4.6, receiving_currency="US Dollar",
        same_currency_amount_delta=0, payment_format="ACH", is_cross_bank=1, is_currency_conversion=0,
        is_self_transfer=0, hour=10, day_of_week=1, day_of_month=1
    )
    
    # Identical timestamp
    e2 = e1.model_copy(update={"event_id": "e2", "transaction_id": 2, "amount_paid": 200.0, "amount_received": 200.0, "receiver_id": "C-1"})
    
    res = engine.process_batch([e1, e2])
    assert len(res) == 2
    
    # Neither should see each other because they are in the same batch
    assert res[0].features["sender_tx_count_1h"] == 0.0
    assert res[1].features["sender_tx_count_1h"] == 0.0
    assert res[0].features["sender_has_no_prior_activity"] == 1.0
    
    # Process subsequent timestamp
    e3 = e1.model_copy(update={"event_id": "e3", "transaction_id": 3, "timestamp": "2023-01-01T10:05:00Z", "amount_paid": 50.0})
    res2 = engine.process_batch([e3])
    
    assert res2[0].features["sender_tx_count_1h"] == 2.0
    assert res2[0].features["sender_amount_sum_1h"] == 300.0
    
def test_engine_nan_semantics():
    engine = OnlineFeatureEngine()
    
    e1 = TransactionEvent(
        event_id="e1", transaction_id=1, timestamp="2023-01-01T10:00:00Z",
        from_bank="A", from_account="1", sender_id="A-1", to_bank="B", to_account="1", receiver_id="B-1",
        amount_paid=100.0, amount_paid_log=4.6, payment_currency="US Dollar", amount_received=100.0, amount_received_log=4.6, 
        receiving_currency="US Dollar", same_currency_amount_delta=0, payment_format="ACH", is_cross_bank=1, 
        is_currency_conversion=0, is_self_transfer=0, hour=10, day_of_week=1, day_of_month=1
    )
    
    res = engine.process_batch([e1])
    f = res[0].features
    
    assert math.isnan(f["sender_seconds_since_previous_activity"])
    assert math.isnan(f["sender_amount_mean_24h"])