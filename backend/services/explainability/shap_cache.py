from collections import OrderedDict
from typing import Optional
from backend.core.schemas import ShapExplanation

class ShapCache:
    """Thread-safe LRU Cache for SHAP explanations to prevent expensive re-computation on dashboard reloads."""
    def __init__(self, maxsize: int = 1000):
        self.cache: OrderedDict[int, ShapExplanation] = OrderedDict()
        self.maxsize = maxsize

    def get(self, transaction_id: int) -> Optional[ShapExplanation]:
        if transaction_id in self.cache:
            self.cache.move_to_end(transaction_id)
            return self.cache[transaction_id]
        return None

    def put(self, transaction_id: int, explanation: ShapExplanation) -> None:
        self.cache[transaction_id] = explanation
        self.cache.move_to_end(transaction_id)
        if len(self.cache) > self.maxsize:
            self.cache.popitem(last=False)