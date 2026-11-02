import math
from datetime import datetime
from typing import Dict, List, Any, Tuple
from collections import defaultdict

from backend.core.schemas import TransactionEvent, FeatureRow
from backend.services.features.account_state import AccountState, PairState

# Time windows mapping to seconds
WINDOWS_SEC = {
    "10m": 600.0,
    "1h": 3600.0,
    "6h": 21600.0,
    "24h": 86400.0,
    "3d": 259200.0
}
EPSILON = 1e-9

def parse_iso_timestamp(ts_str: str) -> float:
    return datetime.fromisoformat(ts_str.replace("Z", "+00:00")).timestamp()

def group_by_timestamp(events: List[TransactionEvent]) -> List[List[TransactionEvent]]:
    """Group sequential events by identical timestamps. Must be pre-sorted."""
    if not events:
        return []
        
    batches = []
    current_batch = [events[0]]
    current_ts = events[0].timestamp
    
    for event in events[1:]:
        if event.timestamp == current_ts:
            current_batch.append(event)
        else:
            batches.append(current_batch)
            current_batch = [event]
            current_ts = event.timestamp
            
    if current_batch:
        batches.append(current_batch)
    return batches

class OnlineFeatureEngine:
    def __init__(self):
        self.accounts: Dict[str, AccountState] = {}
        self.pairs: Dict[Tuple[str, str], PairState] = {}
        
    def get_account(self, account_id: str) -> AccountState:
        if account_id not in self.accounts:
            self.accounts[account_id] = AccountState(account_id)
        return self.accounts[account_id]
        
    def get_pair(self, sender_id: str, receiver_id: str) -> PairState:
        key = (sender_id, receiver_id)
        if key not in self.pairs:
            self.pairs[key] = PairState()
        return self.pairs[key]
        
    def _extract_history_features(self, state: AccountState, current_ts: float, prefix: str) -> Dict[str, float]:
        features: Dict[str, float] = {}
        
        # General activity
        features[f"{prefix}_prior_activity_count"] = float(state.cumulative_events)
        features[f"{prefix}_has_no_prior_activity"] = 1.0 if state.cumulative_events == 0 else 0.0
        
        # If the account has never been seen, its age is exactly 0.0 right now
        if state.first_seen_ts is not None:
            age_sec = current_ts - state.first_seen_ts
        else:
            age_sec = 0.0
            
        age_hours = age_sec / 3600.0
        features[f"{prefix}_observed_age_hours"] = age_hours
        features[f"{prefix}_observed_age_days"] = age_sec / 86400.0
        
        features[f"{prefix}_is_new_1h"] = 1.0 if age_hours < 1.0 else 0.0
        features[f"{prefix}_is_new_24h"] = 1.0 if age_hours < 24.0 else 0.0
        features[f"{prefix}_is_new_3d"] = 1.0 if age_hours < 72.0 else 0.0

        if state.last_event_ts is not None:
            features[f"{prefix}_seconds_since_previous_activity"] = current_ts - state.last_event_ts
        else:
            features[f"{prefix}_seconds_since_previous_activity"] = math.nan

        # Directional
        is_sender = (prefix == "sender")
        dir_count = state.outgoing_cumulative_events if is_sender else state.incoming_cumulative_events
        dir_last_ts = state.outgoing_last_ts if is_sender else state.incoming_last_ts
        
        dir_name = "outgoing" if is_sender else "incoming"
        features[f"{prefix}_{dir_name}_prior_count"] = float(dir_count)
        features[f"{prefix}_{dir_name}_seconds_since_previous"] = (
            (current_ts - dir_last_ts) if dir_last_ts is not None else math.nan
        )

        # Volumes & Velocity
        vol_tracker = state.sender_volume if is_sender else state.receiver_volume
        for suffix in ["10m", "1h", "6h", "24h", "3d"]:
            win_sec = WINDOWS_SEC[suffix]
            cnt, s, sq = vol_tracker.get_stats(current_ts, win_sec)
            features[f"{prefix}_tx_count_{suffix}"] = float(cnt)
            # Volume sums only for 1h, 6h, 24h, 3d
            if suffix != "10m":
                features[f"{prefix}_amount_sum_{suffix}"] = float(s)
                # Internal helper stats required for behavioral deviations
                features[f"{prefix}_amount_sumsq_{suffix}"] = float(sq)
                features[f"{prefix}_amount_count_{suffix}"] = float(cnt)
                
        # Counterparties
        cp_tracker = state.sender_counterparties if is_sender else state.receiver_counterparties
        cp_feature_name = f"unique_receivers" if is_sender else f"unique_senders"
        for suffix in ["1h", "24h"]:
            features[f"{prefix}_{cp_feature_name}_{suffix}"] = float(cp_tracker.get_unique_count(current_ts, WINDOWS_SEC[suffix]))

        # Flow
        for suffix in ["1h", "6h", "24h"]:
            win_sec = WINDOWS_SEC[suffix]
            _, in_sum, _ = state.flow_incoming.get_stats(current_ts, win_sec)
            _, out_sum, _ = state.flow_outgoing.get_stats(current_ts, win_sec)
            features[f"{prefix}_recent_incoming_amount_{suffix}"] = float(in_sum)
            features[f"{prefix}_recent_outgoing_amount_{suffix}"] = float(out_sum)

        return features

    def _apply_deviations(self, f: Dict[str, float], prefix: str, amount_key: str):
        c24 = f.get(f"{prefix}_amount_count_24h", 0.0)
        s24 = f.get(f"{prefix}_amount_sum_24h", 0.0)
        sq24 = f.get(f"{prefix}_amount_sumsq_24h", 0.0)
        amount = f[amount_key]
        
        # Mean
        mean_24h = s24 / c24 if c24 > 0 else math.nan
        f[f"{prefix}_amount_mean_24h"] = mean_24h
        
        # Variance & Std
        var_24h = (sq24 / c24 - mean_24h * mean_24h) if c24 > 1 else math.nan
        std_24h = math.sqrt(var_24h) if not math.isnan(var_24h) and var_24h >= 0 else 0.0
        f[f"{prefix}_amount_std_24h"] = std_24h
        
        # Ratios
        f[f"{prefix}_amount_ratio_to_mean_24h"] = (amount / mean_24h) if not math.isnan(mean_24h) and mean_24h > 0 else math.nan
        
        if not math.isnan(std_24h) and std_24h > 0:
            f[f"{prefix}_amount_zscore_24h"] = (amount - mean_24h) / std_24h
        else:
            f[f"{prefix}_amount_zscore_24h"] = math.nan

    def _apply_flow_ratios(self, f: Dict[str, float], prefix: str):
        for suffix in ["1h", "24h"]:
            inc = f.get(f"{prefix}_recent_incoming_amount_{suffix}", 0.0)
            outg = f.get(f"{prefix}_recent_outgoing_amount_{suffix}", 0.0)
            
            f[f"{prefix}_inflow_outflow_ratio_{suffix}"] = inc / (outg + EPSILON)
            f[f"{prefix}_flow_balance_{suffix}"] = (inc - outg) / (inc + outg + EPSILON)
            f[f"{prefix}_pass_through_ratio_{suffix}"] = min(inc, outg) / (max(inc, outg) + EPSILON)
            
            if inc > 0:
                f[f"{prefix}_fund_retention_{suffix}"] = (inc - outg) / (inc + EPSILON)
            else:
                f[f"{prefix}_fund_retention_{suffix}"] = math.nan

    def process_batch(self, events: List[TransactionEvent]) -> List[FeatureRow]: # type: ignore
        """
        Process a batch of events with strictly identical timestamps.
        Computes features from pre-batch state, then applies updates.
        """
        if not events:
            return []
            
        current_ts_str = events[0].timestamp
        current_ts = parse_iso_timestamp(current_ts_str)
        
        for e in events:
            if e.timestamp != current_ts_str:
                raise ValueError("OnlineFeatureEngine.process_batch requires all events to have identical timestamps.")
                
        # 1. Prune necessary account states FIRST for this batch timestamp
        accounts_to_prune = set()
        for e in events:
            accounts_to_prune.add(e.sender_id)
            accounts_to_prune.add(e.receiver_id)
            
        for acc_id in accounts_to_prune:
            self.get_account(acc_id).prune_windows(current_ts)
            
        # 2. Extract features before any state is mutated
        results: List[FeatureRow] = []
        for e in events:
            sender_state = self.get_account(e.sender_id)
            receiver_state = self.get_account(e.receiver_id)
            pair_state = self.get_pair(e.sender_id, e.receiver_id)
            
            f: Dict[str, float | str] = {}
            
            # Static & Base Features
            f["amount_paid"] = float(e.amount_paid)
            f["amount_paid_log"] = float(e.amount_paid_log)
            f["amount_received"] = float(e.amount_received)
            f["amount_received_log"] = float(e.amount_received_log)
            f["same_currency_amount_delta"] = float(e.same_currency_amount_delta) if e.same_currency_amount_delta is not None else math.nan
            f["is_cross_bank"] = float(e.is_cross_bank)
            f["is_currency_conversion"] = float(e.is_currency_conversion)
            f["is_self_transfer"] = float(e.is_self_transfer)
            f["hour"] = float(e.hour)
            f["day_of_week"] = float(e.day_of_week)
            f["day_of_month"] = float(e.day_of_month)
            
            f["hour_sin"] = math.sin(2 * math.pi * e.hour / 24.0)
            f["hour_cos"] = math.cos(2 * math.pi * e.hour / 24.0)
            f["day_of_week_sin"] = math.sin(2 * math.pi * e.day_of_week / 7.0)
            f["day_of_week_cos"] = math.cos(2 * math.pi * e.day_of_week / 7.0)
            
            # Categoricals (passed through for one-hot encoding later)
            f["payment_format"] = e.payment_format
            f["payment_currency"] = e.payment_currency
            f["receiving_currency"] = e.receiving_currency
            
            # Account History Features
            s_hist = self._extract_history_features(sender_state, current_ts, "sender")
            r_hist = self._extract_history_features(receiver_state, current_ts, "receiver")
            f.update(s_hist)
            f.update(r_hist)
            
            # Pair Features
            f["sender_receiver_prior_tx_count"] = float(pair_state.prior_tx_count)
            f["is_new_counterparty"] = 1.0 if pair_state.prior_tx_count == 0 else 0.0
            
            # Deviations and Ratios
            self._apply_deviations(f, "sender", "amount_paid") # type: ignore
            self._apply_deviations(f, "receiver", "amount_received") # type: ignore
            self._apply_flow_ratios(f, "sender") # type: ignore
            self._apply_flow_ratios(f, "receiver") # type: ignore
            
            results.append(FeatureRow(transaction_id=e.transaction_id, features=f)) # type: ignore

        # 3. Apply state mutations
        # Track same-timestamp deduplication for account flow
        flow_updates = defaultdict(lambda: {"in": 0.0, "out": 0.0})
        # Track overall account activity to collapse identical timestamps
        activity_seen = set()
        
        for e in events:
            ss = self.get_account(e.sender_id)
            rs = self.get_account(e.receiver_id)
            ps = self.get_pair(e.sender_id, e.receiver_id)
            
            # Pair update
            ps.prior_tx_count += 1
            
            # Volumes & Counterparties
            ss.sender_volume.add(current_ts, e.amount_paid)
            ss.sender_counterparties.add(current_ts, e.receiver_id)
            
            rs.receiver_volume.add(current_ts, e.amount_received)
            rs.receiver_counterparties.add(current_ts, e.sender_id)
            
            # Account Flow Updates
            flow_updates[e.sender_id]["out"] += e.amount_paid
            flow_updates[e.receiver_id]["in"] += e.amount_received
            
            # Activity Tracking
            activity_seen.add(e.sender_id)
            activity_seen.add(e.receiver_id)
            
            # Directional Events Update
            ss.outgoing_cumulative_events += 1
            ss.outgoing_last_ts = current_ts
            
            rs.incoming_cumulative_events += 1
            rs.incoming_last_ts = current_ts
            
        # Apply collapsed flow and general activity
        for acc_id in activity_seen:
            acc_state = self.get_account(acc_id)
            if acc_state.first_seen_ts is None:
                acc_state.first_seen_ts = current_ts
                
            acc_state.last_event_ts = current_ts
            acc_state.cumulative_events += 1
            
            # Apply aggregated flow for this timestamp
            f_in = flow_updates[acc_id]["in"]
            f_out = flow_updates[acc_id]["out"]
            if f_in > 0 or acc_id in flow_updates:  # even if 0, record flow event
                acc_state.flow_incoming.add(current_ts, f_in)
                acc_state.flow_outgoing.add(current_ts, f_out)

        return results
    
    def warmup_batch(self, events: List[TransactionEvent]) -> None:
        """
        Optimized state mutation for warm-starting.
        Skips all dictionary allocation and feature math.
        Requires identical timestamps in the batch.
        """
        if not events:
            return
            
        current_ts_str = events[0].timestamp
        current_ts = parse_iso_timestamp(current_ts_str)
        
        # 1. Prune necessary account states FIRST
        accounts_to_prune = set()
        for e in events:
            accounts_to_prune.add(e.sender_id)
            accounts_to_prune.add(e.receiver_id)
            
        for acc_id in accounts_to_prune:
            self.get_account(acc_id).prune_windows(current_ts)
            
        # 2. Apply state mutations (skipping extraction)
        from collections import defaultdict
        flow_updates = defaultdict(lambda: {"in": 0.0, "out": 0.0})
        activity_seen = set()
        
        for e in events:
            ss = self.get_account(e.sender_id)
            rs = self.get_account(e.receiver_id)
            ps = self.get_pair(e.sender_id, e.receiver_id)
            
            # Pair update
            ps.prior_tx_count += 1
            
            # Volumes & Counterparties
            ss.sender_volume.add(current_ts, e.amount_paid)
            ss.sender_counterparties.add(current_ts, e.receiver_id)
            
            rs.receiver_volume.add(current_ts, e.amount_received)
            rs.receiver_counterparties.add(current_ts, e.sender_id)
            
            # Account Flow Updates
            flow_updates[e.sender_id]["out"] += e.amount_paid
            flow_updates[e.receiver_id]["in"] += e.amount_received
            
            # Activity Tracking
            activity_seen.add(e.sender_id)
            activity_seen.add(e.receiver_id)
            
            # Directional Events Update
            ss.outgoing_cumulative_events += 1
            ss.outgoing_last_ts = current_ts
            
            rs.incoming_cumulative_events += 1
            rs.incoming_last_ts = current_ts
            
        # Apply collapsed flow and general activity
        for acc_id in activity_seen:
            acc_state = self.get_account(acc_id)
            if acc_state.first_seen_ts is None:
                acc_state.first_seen_ts = current_ts
                
            acc_state.last_event_ts = current_ts
            acc_state.cumulative_events += 1
            
            f_in = flow_updates[acc_id]["in"]
            f_out = flow_updates[acc_id]["out"]
            if f_in > 0 or acc_id in flow_updates: 
                acc_state.flow_incoming.add(current_ts, f_in)
                acc_state.flow_outgoing.add(current_ts, f_out)
