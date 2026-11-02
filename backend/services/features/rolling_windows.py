from collections import deque
from typing import Tuple, List, Set

class RollingStatWindow:
    """
    Maintains a chronological queue of events (timestamp, amount) up to 
    a maximum window size. Can answer sum, sum of squares, and count 
    for any bounded window size <= max_window strictly.
    """
    __slots__ = ["max_window", "events"]
    
    def __init__(self, max_window: float = 259200.0): # default 3 days
        self.max_window = max_window
        # (timestamp, amount)
        self.events: deque[Tuple[float, float]] = deque()
        
    def add(self, ts: float, amount: float) -> None:
        self.events.append((ts, amount))
        
    def prune(self, current_ts: float) -> None:
        """Drop events strictly older than max_window."""
        while self.events and current_ts - self.events[0][0] > self.max_window:
            self.events.popleft()
            
    def get_stats(self, current_ts: float, window: float) -> Tuple[int, float, float]:
        """
        Returns (count, sum, sum_squares) for the specified rolling window [current_ts - window, current_ts).
        Relies on the fact that events inside the queue are ALREADY strictly < current_ts.
        """
        count = 0
        total_sum = 0.0
        total_sumsq = 0.0
        
        # Walk backwards for speed (most queries are shorter than max_window)
        for i in range(len(self.events) - 1, -1, -1):
            ts, amount = self.events[i]
            if current_ts - ts > window:
                break
            count += 1
            total_sum += amount
            total_sumsq += amount * amount
            
        return count, total_sum, total_sumsq


class RollingUniqueWindow:
    """
    Maintains a chronological queue of counterparties (timestamp, ID) to 
    compute unique counterparties over a bounded window.
    """
    __slots__ = ["max_window", "events"]
    
    def __init__(self, max_window: float = 86400.0): # default 24 hours
        self.max_window = max_window
        # (timestamp, counterparty_id)
        self.events: deque[Tuple[float, str]] = deque()
        
    def add(self, ts: float, counterparty_id: str) -> None:
        self.events.append((ts, counterparty_id))
        
    def prune(self, current_ts: float) -> None:
        while self.events and current_ts - self.events[0][0] > self.max_window:
            self.events.popleft()
            
    def get_unique_count(self, current_ts: float, window: float) -> int:
        unique_seen: Set[str] = set()
        
        for i in range(len(self.events) - 1, -1, -1):
            ts, cp_id = self.events[i]
            if current_ts - ts > window:
                break
            unique_seen.add(cp_id)
            
        return len(unique_seen)