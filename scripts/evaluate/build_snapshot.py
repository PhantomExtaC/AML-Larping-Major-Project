from pathlib import Path
from backend.services.features.online_feature_engine import OnlineFeatureEngine
from backend.services.features.snapshot import warm_start, save_snapshot

def main():
    base_dir = Path(__file__).resolve().parents[2]
    train_file = base_dir / "data/processed/hi_small/modeling/splits/train.parquet"
    out_file = base_dir / "data/processed/hi_small/modeling/splits/train_snapshot.pkl"
    
    engine = OnlineFeatureEngine()
    print(f"Warming up engine using historical data: {train_file.name}")
    
    # Process the entire train split to build account states (no scoring)
    warm_start(engine, str(train_file))
    
    print(f"Saving engine state to {out_file.name}")
    save_snapshot(engine, out_file)
    print("Snapshot ready.")

if __name__ == "__main__":
    main()