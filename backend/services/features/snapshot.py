import pickle
import polars as pl
from pathlib import Path
from tqdm import tqdm
from backend.services.features.online_feature_engine import OnlineFeatureEngine, parse_iso_timestamp, group_by_timestamp
from backend.core.schemas import TransactionEvent

def save_snapshot(engine: OnlineFeatureEngine, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"accounts": engine.accounts, "pairs": engine.pairs}, f)

def load_snapshot(path: Path) -> OnlineFeatureEngine:
    with open(path, "rb") as f:
        state = pickle.load(f)
    engine = OnlineFeatureEngine()
    engine.accounts = state["accounts"]
    engine.pairs = state["pairs"]
    return engine

def warm_start(engine: OnlineFeatureEngine, parquet_pattern: str, up_to_timestamp: str | None = None) -> None:
    """
    Replays transaction events directly into the state without generating features.
    Used to quickly populate account histories from prior offline splits.
    """
    # Load data and format timestamps as strict ISO strings
    df = (
        pl.scan_parquet(parquet_pattern)
        .with_columns(pl.col("timestamp").dt.to_string("%Y-%m-%dT%H:%M:%SZ"))
        .sort(["timestamp", "transaction_id"])
        .collect(engine="streaming")
    )
    
    cutoff = None
    if up_to_timestamp:
        cutoff = parse_iso_timestamp(up_to_timestamp)
        
    events = []
    for row in tqdm(df.to_dicts(), desc="Warm starting Engine"):
        if cutoff is not None and parse_iso_timestamp(row["timestamp"]) >= cutoff:
            break
        # Inject the required event_id missing from raw parquet
        events.append(TransactionEvent(event_id=f"warmup_{row['transaction_id']}", **row))
        
    batches = group_by_timestamp(events)
    for batch in tqdm(batches, desc="Applying Batches"):
        engine.warmup_batch(batch) # Calls the new optimized method