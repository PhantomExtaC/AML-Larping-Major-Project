from pathlib import Path
import sys

import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "hi_small"
    / "modeling"
    / "phase2a"
    / "features.parquet"
)


def main():

    if len(sys.argv) < 2:
        print(
            "Usage:\n"
            "python scripts/data/"
            "inspect_account_history.py "
            "<account_id>"
        )
        return

    account_id = sys.argv[1]

    df = (
        pl.scan_parquet(
            FEATURE_FILE
        )
        .filter(
            (pl.col("sender_id") == account_id)
            |
            (pl.col("receiver_id") == account_id)
        )
        .sort(
            [
                "timestamp",
                "transaction_id",
            ]
        )
        .select(
            [
                "transaction_id",
                "timestamp",

                "sender_id",
                "receiver_id",

                "sender_prior_activity_count",
                "receiver_prior_activity_count",

                "sender_observed_age_hours",
                "receiver_observed_age_hours",

                "sender_seconds_since_previous_activity",
                "receiver_seconds_since_previous_activity",
            ]
        )
        .collect()
    )

    print(df)


if __name__ == "__main__":
    main()