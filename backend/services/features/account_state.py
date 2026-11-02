from typing import Optional
from backend.services.features.rolling_windows import RollingStatWindow, RollingUniqueWindow

class PairState:
    """Tracks state between a specific sender and receiver pair."""
    __slots__ = ["prior_tx_count"]
    
    def __init__(self):
        self.prior_tx_count: int = 0

class AccountState:
    """
    Unified state for an account, handling both sender and receiver roles.
    Matches offline chronological bounds and exact NaN semantics.
    """
    __slots__ = [
        "account_id", "first_seen_ts", "last_event_ts", "cumulative_events",
        "outgoing_cumulative_events", "outgoing_last_ts",
        "incoming_cumulative_events", "incoming_last_ts",
        "sender_volume", "receiver_volume", "flow_incoming", "flow_outgoing",
        "sender_counterparties", "receiver_counterparties"
    ]
    
    def __init__(self, account_id: str):
        self.account_id = account_id
        
        # General activity
        self.first_seen_ts: Optional[float] = None
        self.last_event_ts: Optional[float] = None
        self.cumulative_events: int = 0
        
        # Directional activity
        self.outgoing_cumulative_events: int = 0
        self.outgoing_last_ts: Optional[float] = None
        
        self.incoming_cumulative_events: int = 0
        self.incoming_last_ts: Optional[float] = None
        
        # Rolling volumes (amount paid / amount received) - max 3 days
        self.sender_volume = RollingStatWindow(max_window=259200.0)
        self.receiver_volume = RollingStatWindow(max_window=259200.0)
        
        # Account flow (sum of all incoming / outgoing) - max 24 hours
        self.flow_incoming = RollingStatWindow(max_window=86400.0)
        self.flow_outgoing = RollingStatWindow(max_window=86400.0)
        
        # Counterparties - max 24 hours
        self.sender_counterparties = RollingUniqueWindow(max_window=86400.0)
        self.receiver_counterparties = RollingUniqueWindow(max_window=86400.0)

    def prune_windows(self, current_ts: float):
        """Clean up all internal deques for the current timestamp."""
        self.sender_volume.prune(current_ts)
        self.receiver_volume.prune(current_ts)
        self.flow_incoming.prune(current_ts)
        self.flow_outgoing.prune(current_ts)
        self.sender_counterparties.prune(current_ts)
        self.receiver_counterparties.prune(current_ts)