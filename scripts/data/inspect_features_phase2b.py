from __future__ import annotations

import sys
from pathlib import Path

import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "hi_small"
    / "modeling"
    / "phase2b"
    / "features.parquet"
)


def main() -> None:

    if len(sys.argv) < 2:
        print(
            "Usage:\n"
            "python scripts/data/"
            "inspect_features_phase2b.py "
            "<account_id>"
        )
        return

    account_id = (
        sys.argv[1]
    )

    lf = (
        pl.scan_parquet(
            str(FEATURE_FILE)
        )
    )

    df = (
        lf
        .filter(
            (
                pl.col(
                    "sender_id"
                )
                == account_id
            )
            |
            (
                pl.col(
                    "receiver_id"
                )
                == account_id
            )
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

                "amount_paid",

                "sender_tx_count_10m",
                "sender_tx_count_1h",
                "sender_tx_count_24h",

                "receiver_tx_count_10m",
                "receiver_tx_count_1h",
                "receiver_tx_count_24h",

                "sender_unique_receivers_1h",
                "sender_unique_receivers_24h",

                "receiver_unique_senders_1h",
                "receiver_unique_senders_24h",

                "sender_receiver_prior_tx_count",
                "is_new_counterparty",

                "sender_amount_mean_24h",
                "sender_amount_ratio_to_mean_24h",
                "sender_amount_zscore_24h",

                "sender_recent_incoming_amount_1h",
                "sender_recent_outgoing_amount_1h",

                "sender_flow_balance_1h",
                "sender_pass_through_ratio_1h",

                "is_laundering",
            ]
        )
        .collect(
            engine="streaming"
        )
    )

    pl.Config.set_tbl_rows(
        100
    )

    pl.Config.set_tbl_cols(
        30
    )

    pl.Config.set_tbl_width_chars(
        220
    )

    print(
        df
    )


if __name__ == "__main__":
    main()