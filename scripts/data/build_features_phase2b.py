from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import polars as pl


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PHASE2A_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "hi_small"
    / "modeling"
    / "phase2a"
    / "features.parquet"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "hi_small"
    / "modeling"
    / "phase2b"
)

STAGE_DIR = OUTPUT_DIR / "stages"

STAGE1_FILE = (
    STAGE_DIR
    / "01_velocity_volume.parquet"
)

STAGE2_FILE = (
    STAGE_DIR
    / "02_counterparty.parquet"
)

STAGE3_FILE = (
    STAGE_DIR
    / "03_flow_deviation.parquet"
)

FINAL_FILE = (
    OUTPUT_DIR
    / "features.parquet"
)

METADATA_FILE = (
    OUTPUT_DIR
    / "feature_metadata.json"
)

EXPECTED_ROWS = 5_078_345
EXPECTED_LAUNDERING = 5_177


# Rolling windows used by the baseline.
COUNT_WINDOWS = {
    "10m": "10m",
    "1h": "1h",
    "6h": "6h",
    "24h": "24h",
    "3d": "3d",
}

VOLUME_WINDOWS = {
    "1h": "1h",
    "6h": "6h",
    "24h": "24h",
    "3d": "3d",
}

COUNTERPARTY_WINDOWS = {
    "1h": "1h",
    "24h": "24h",
}

FLOW_WINDOWS = {
    "1h": "1h",
    "6h": "6h",
    "24h": "24h",
}

EPSILON = 1e-9


# ============================================================
# General helpers
# ============================================================

def print_step(message: str) -> None:
    print("\n" + "=" * 76)
    print(message)
    print("=" * 76)


def ensure_directories() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    STAGE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def validate_inputs() -> None:
    if not PHASE2A_FILE.exists():
        raise FileNotFoundError(
            "Phase 2A feature file was not found:\n"
            f"{PHASE2A_FILE}"
        )


def scan(path: Path) -> pl.LazyFrame:
    return pl.scan_parquet(
        str(path)
    )


def sink(
    frame: pl.LazyFrame,
    path: Path,
) -> None:
    """
    Execute a lazy plan using the streaming engine and write
    directly to Parquet.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if path.exists():
        path.unlink()

    frame.sink_parquet(
        str(path),
        compression="zstd",
        statistics=True,
        maintain_order=True,
        engine="streaming",
    )


def resolve_schema(
    frame: pl.LazyFrame,
    stage_name: str,
) -> None:

    print_step(
        f"Schema check — {stage_name}"
    )

    schema = frame.collect_schema()

    print(
        f"Columns: {len(schema.names())}"
    )

    for name, dtype in schema.items():
        print(
            f"{name:<60} {dtype}"
        )


def verify_base_invariants(
    path: Path,
) -> None:
    """
    Cheap post-stage validation.
    """

    summary = (
        scan(path)
        .select(
            [
                pl.len()
                .alias("rows"),

                pl.col(
                    "transaction_id"
                )
                .n_unique()
                .alias(
                    "unique_transactions"
                ),

                pl.col(
                    "is_laundering"
                )
                .sum()
                .alias(
                    "laundering"
                ),
            ]
        )
        .collect(
            engine="streaming"
        )
        .to_dicts()[0]
    )

    if summary["rows"] != EXPECTED_ROWS:
        raise ValueError(
            f"Row count changed: {summary}"
        )

    if (
        summary["unique_transactions"]
        != EXPECTED_ROWS
    ):
        raise ValueError(
            "Transaction IDs are no longer unique."
        )

    if (
        summary["laundering"]
        != EXPECTED_LAUNDERING
    ):
        raise ValueError(
            "Target labels changed."
        )

    print(
        "Invariant check passed: "
        f"{summary}"
    )


# ============================================================
# Generic rolling helper
# ============================================================

def rolling_timestamp_features(
    timestamp_table: pl.LazyFrame,
    group_column: str,
    window: str,
    aggregations: list[pl.Expr],
) -> pl.LazyFrame:
    """
    Build a rolling historical table.

    closed='left' means the historical window is:

        [t - window, t)

    therefore events at the current timestamp are excluded.
    """

    ordered = (
        timestamp_table
        .sort(
            [
                group_column,
                "timestamp",
            ]
        )
    )

    return (
        ordered
        .rolling(
            index_column="timestamp",
            period=window,
            closed="left",
            group_by=group_column,
        )
        .agg(
            aggregations
        )
    )


# ============================================================
# STAGE 1
# Velocity + monetary volume
# ============================================================

def build_sender_timestamp_table(
    base: pl.LazyFrame,
) -> pl.LazyFrame:

    return (
        base
        .select(
            [
                "sender_id",
                "timestamp",
                "amount_paid",
            ]
        )
        .group_by(
            [
                "sender_id",
                "timestamp",
            ]
        )
        .agg(
            [
                pl.len()
                .alias(
                    "sender_events_at_timestamp"
                ),

                pl.col(
                    "amount_paid"
                )
                .sum()
                .alias(
                    "sender_amount_at_timestamp"
                ),

                (
                    pl.col(
                        "amount_paid"
                    )
                    * pl.col(
                        "amount_paid"
                    )
                )
                .sum()
                .alias(
                    "sender_amount_sq_at_timestamp"
                ),
            ]
        )
    )


def build_receiver_timestamp_table(
    base: pl.LazyFrame,
) -> pl.LazyFrame:

    return (
        base
        .select(
            [
                "receiver_id",
                "timestamp",
                "amount_received",
            ]
        )
        .group_by(
            [
                "receiver_id",
                "timestamp",
            ]
        )
        .agg(
            [
                pl.len()
                .alias(
                    "receiver_events_at_timestamp"
                ),

                pl.col(
                    "amount_received"
                )
                .sum()
                .alias(
                    "receiver_amount_at_timestamp"
                ),

                (
                    pl.col(
                        "amount_received"
                    )
                    * pl.col(
                        "amount_received"
                    )
                )
                .sum()
                .alias(
                    "receiver_amount_sq_at_timestamp"
                ),
            ]
        )
    )


def add_sender_rolling_features(
    base: pl.LazyFrame,
) -> pl.LazyFrame:

    sender_ts = (
        build_sender_timestamp_table(
            base
        )
    )

    result = base

    # --------------------------------------------------------
    # Counts
    # --------------------------------------------------------

    for suffix, period in COUNT_WINDOWS.items():

        rolling = (
            rolling_timestamp_features(
                sender_ts,
                group_column="sender_id",
                window=period,
                aggregations=[
                    pl.col(
                        "sender_events_at_timestamp"
                    )
                    .sum()
                    .alias(
                        f"sender_tx_count_{suffix}"
                    )
                ],
            )
        )

        result = (
            result
            .join(
                rolling,
                on=[
                    "sender_id",
                    "timestamp",
                ],
                how="left",
            )
        )

    # --------------------------------------------------------
    # Monetary volume
    # --------------------------------------------------------

    for suffix, period in VOLUME_WINDOWS.items():

        rolling = (
            rolling_timestamp_features(
                sender_ts,
                group_column="sender_id",
                window=period,
                aggregations=[
                    pl.col(
                        "sender_amount_at_timestamp"
                    )
                    .sum()
                    .alias(
                        f"sender_amount_sum_{suffix}"
                    ),

                    pl.col(
                        "sender_amount_sq_at_timestamp"
                    )
                    .sum()
                    .alias(
                        f"sender_amount_sumsq_{suffix}"
                    ),

                    pl.col(
                        "sender_events_at_timestamp"
                    )
                    .sum()
                    .alias(
                        f"sender_amount_count_{suffix}"
                    ),
                ],
            )
        )

        result = (
            result
            .join(
                rolling,
                on=[
                    "sender_id",
                    "timestamp",
                ],
                how="left",
            )
        )

    return result


def add_receiver_rolling_features(
    base: pl.LazyFrame,
) -> pl.LazyFrame:

    receiver_ts = (
        build_receiver_timestamp_table(
            base
        )
    )

    result = base

    for suffix, period in COUNT_WINDOWS.items():

        rolling = (
            rolling_timestamp_features(
                receiver_ts,
                group_column="receiver_id",
                window=period,
                aggregations=[
                    pl.col(
                        "receiver_events_at_timestamp"
                    )
                    .sum()
                    .alias(
                        f"receiver_tx_count_{suffix}"
                    )
                ],
            )
        )

        result = (
            result
            .join(
                rolling,
                on=[
                    "receiver_id",
                    "timestamp",
                ],
                how="left",
            )
        )

    for suffix, period in VOLUME_WINDOWS.items():

        rolling = (
            rolling_timestamp_features(
                receiver_ts,
                group_column="receiver_id",
                window=period,
                aggregations=[
                    pl.col(
                        "receiver_amount_at_timestamp"
                    )
                    .sum()
                    .alias(
                        f"receiver_amount_sum_{suffix}"
                    ),

                    pl.col(
                        "receiver_amount_sq_at_timestamp"
                    )
                    .sum()
                    .alias(
                        f"receiver_amount_sumsq_{suffix}"
                    ),

                    pl.col(
                        "receiver_events_at_timestamp"
                    )
                    .sum()
                    .alias(
                        f"receiver_amount_count_{suffix}"
                    ),
                ],
            )
        )

        result = (
            result
            .join(
                rolling,
                on=[
                    "receiver_id",
                    "timestamp",
                ],
                how="left",
            )
        )

    return result


def fill_stage1_history_nulls(
    frame: pl.LazyFrame,
) -> pl.LazyFrame:

    count_columns = []

    for suffix in COUNT_WINDOWS:
        count_columns.extend(
            [
                f"sender_tx_count_{suffix}",
                f"receiver_tx_count_{suffix}",
            ]
        )

    for suffix in VOLUME_WINDOWS:
        count_columns.extend(
            [
                f"sender_amount_sum_{suffix}",
                f"receiver_amount_sum_{suffix}",

                f"sender_amount_sumsq_{suffix}",
                f"receiver_amount_sumsq_{suffix}",

                f"sender_amount_count_{suffix}",
                f"receiver_amount_count_{suffix}",
            ]
        )

    return (
        frame
        .with_columns(
            [
                pl.col(column)
                .fill_null(0)
                for column
                in count_columns
            ]
        )
    )


def stage1_velocity_volume() -> None:

    print_step(
        "PHASE 2B — Stage 1: Velocity and volume"
    )

    base = scan(
        PHASE2A_FILE
    )

    frame = (
        add_sender_rolling_features(
            base
        )
    )

    frame = (
        add_receiver_rolling_features(
            frame
        )
    )

    frame = (
        fill_stage1_history_nulls(
            frame
        )
    )

    resolve_schema(
        frame,
        "Stage 1",
    )

    sink(
        frame,
        STAGE1_FILE,
    )

    verify_base_invariants(
        STAGE1_FILE
    )

    print(
        f"\nStage 1 written:\n"
        f"{STAGE1_FILE}"
    )


# ============================================================
# STAGE 2
# Counterparty behaviour
# ============================================================

def sender_counterparty_events(
    base: pl.LazyFrame,
) -> pl.LazyFrame:
    """
    One row per sender/receiver relationship occurrence at
    each timestamp.

    Multiple transactions to the same receiver at exactly the
    same timestamp count as one distinct active counterparty.
    """

    return (
        base
        .select(
            [
                "sender_id",
                "receiver_id",
                "timestamp",
            ]
        )
        .unique(
            subset=[
                "sender_id",
                "receiver_id",
                "timestamp",
            ]
        )
    )


def receiver_counterparty_events(
    base: pl.LazyFrame,
) -> pl.LazyFrame:

    return (
        base
        .select(
            [
                "receiver_id",
                "sender_id",
                "timestamp",
            ]
        )
        .unique(
            subset=[
                "receiver_id",
                "sender_id",
                "timestamp",
            ]
        )
    )


def add_rolling_counterparty_features(
    base: pl.LazyFrame,
) -> pl.LazyFrame:

    result = base

    sender_events = (
        sender_counterparty_events(
            base
        )
        .sort(
            [
                "sender_id",
                "timestamp",
            ]
        )
    )

    receiver_events = (
        receiver_counterparty_events(
            base
        )
        .sort(
            [
                "receiver_id",
                "timestamp",
            ]
        )
    )

    for suffix, period in (
        COUNTERPARTY_WINDOWS.items()
    ):

        # ----------------------------------------------------
        # Sender unique receivers
        # ----------------------------------------------------

        sender_roll_raw = (
            sender_events
            .rolling(
                index_column="timestamp",
                period=period,
                closed="left",
                group_by="sender_id",
            )
            .agg(
                pl.col(
                    "receiver_id"
                )
                .n_unique()
                .alias(
                    f"sender_unique_receivers_{suffix}"
                )
            )
        )

        # Multiple pair rows may have the same sender/timestamp.
        # Because current timestamp is excluded, all receive
        # identical historical values. Collapse to one row.
        sender_roll = (
            sender_roll_raw
            .group_by(
                [
                    "sender_id",
                    "timestamp",
                ]
            )
            .agg(
                pl.col(
                    f"sender_unique_receivers_{suffix}"
                )
                .first()
            )
        )

        result = (
            result
            .join(
                sender_roll,
                on=[
                    "sender_id",
                    "timestamp",
                ],
                how="left",
            )
        )

        # ----------------------------------------------------
        # Receiver unique senders
        # ----------------------------------------------------

        receiver_roll_raw = (
            receiver_events
            .rolling(
                index_column="timestamp",
                period=period,
                closed="left",
                group_by="receiver_id",
            )
            .agg(
                pl.col(
                    "sender_id"
                )
                .n_unique()
                .alias(
                    f"receiver_unique_senders_{suffix}"
                )
            )
        )

        receiver_roll = (
            receiver_roll_raw
            .group_by(
                [
                    "receiver_id",
                    "timestamp",
                ]
            )
            .agg(
                pl.col(
                    f"receiver_unique_senders_{suffix}"
                )
                .first()
            )
        )

        result = (
            result
            .join(
                receiver_roll,
                on=[
                    "receiver_id",
                    "timestamp",
                ],
                how="left",
            )
        )

    return result


def build_pair_history(
    base: pl.LazyFrame,
) -> pl.LazyFrame:
    """
    Historical relationship strength:

        How many transactions occurred between this exact
        sender -> receiver pair before the current timestamp?
    """

    pair_table = (
        base
        .select(
            [
                "sender_id",
                "receiver_id",
                "timestamp",
            ]
        )
        .group_by(
            [
                "sender_id",
                "receiver_id",
                "timestamp",
            ]
        )
        .agg(
            pl.len()
            .alias(
                "pair_events_at_timestamp"
            )
        )
        .sort(
            [
                "sender_id",
                "receiver_id",
                "timestamp",
            ]
        )
    )

    pair_table = (
        pair_table
        .with_columns(
            [
                pl.col(
                    "pair_events_at_timestamp"
                )
                .cum_sum()
                .over(
                    [
                        "sender_id",
                        "receiver_id",
                    ]
                )
                .alias(
                    "pair_cumulative_events"
                )
            ]
        )
    )

    pair_table = (
        pair_table
        .with_columns(
            [
                (
                    pl.col(
                        "pair_cumulative_events"
                    )
                    -
                    pl.col(
                        "pair_events_at_timestamp"
                    )
                )
                .cast(
                    pl.Int64
                )
                .alias(
                    "sender_receiver_prior_tx_count"
                )
            ]
        )
        .select(
            [
                "sender_id",
                "receiver_id",
                "timestamp",
                "sender_receiver_prior_tx_count",
            ]
        )
    )

    return pair_table


def stage2_counterparties() -> None:

    print_step(
        "PHASE 2B — Stage 2: Counterparty behaviour"
    )

    base = scan(
        STAGE1_FILE
    )

    frame = (
        add_rolling_counterparty_features(
            base
        )
    )

    pair_history = (
        build_pair_history(
            base
        )
    )

    frame = (
        frame
        .join(
            pair_history,
            on=[
                "sender_id",
                "receiver_id",
                "timestamp",
            ],
            how="left",
        )
    )

    fill_columns = [
        "sender_unique_receivers_1h",
        "sender_unique_receivers_24h",
        "receiver_unique_senders_1h",
        "receiver_unique_senders_24h",
        "sender_receiver_prior_tx_count",
    ]

    frame = (
        frame
        .with_columns(
            [
                pl.col(column)
                .fill_null(0)
                for column
                in fill_columns
            ]
        )
        .with_columns(
            [
                (
                    pl.col(
                        "sender_receiver_prior_tx_count"
                    )
                    == 0
                )
                .cast(
                    pl.Int8
                )
                .alias(
                    "is_new_counterparty"
                )
            ]
        )
    )

    resolve_schema(
        frame,
        "Stage 2",
    )

    sink(
        frame,
        STAGE2_FILE,
    )

    verify_base_invariants(
        STAGE2_FILE
    )

    print(
        f"\nStage 2 written:\n"
        f"{STAGE2_FILE}"
    )


# ============================================================
# STAGE 3
# Account flow + behavioural deviation
# ============================================================

def build_account_flow_table(
    base: pl.LazyFrame,
) -> pl.LazyFrame:
    """
    Convert transaction edges into account-level money flows.

    sender:
        outgoing = amount_paid

    receiver:
        incoming = amount_received
    """

    outgoing = (
        base
        .select(
            [
                pl.col(
                    "sender_id"
                )
                .alias(
                    "account_id"
                ),

                "timestamp",

                pl.col(
                    "amount_paid"
                )
                .alias(
                    "outgoing_amount_at_timestamp"
                ),

                pl.lit(
                    0.0
                )
                .alias(
                    "incoming_amount_at_timestamp"
                ),
            ]
        )
    )

    incoming = (
        base
        .select(
            [
                pl.col(
                    "receiver_id"
                )
                .alias(
                    "account_id"
                ),

                "timestamp",

                pl.lit(
                    0.0
                )
                .alias(
                    "outgoing_amount_at_timestamp"
                ),

                pl.col(
                    "amount_received"
                )
                .alias(
                    "incoming_amount_at_timestamp"
                ),
            ]
        )
    )

    flow = (
        pl.concat(
            [
                outgoing,
                incoming,
            ]
        )
        .group_by(
            [
                "account_id",
                "timestamp",
            ]
        )
        .agg(
            [
                pl.col(
                    "incoming_amount_at_timestamp"
                )
                .sum(),

                pl.col(
                    "outgoing_amount_at_timestamp"
                )
                .sum(),
            ]
        )
    )

    return flow


def build_rolling_flow_table(
    base: pl.LazyFrame,
) -> pl.LazyFrame:

    flow = (
        build_account_flow_table(
            base
        )
        .sort(
            [
                "account_id",
                "timestamp",
            ]
        )
    )

    result = None

    for suffix, period in (
        FLOW_WINDOWS.items()
    ):

        rolling = (
            flow
            .rolling(
                index_column="timestamp",
                period=period,
                closed="left",
                group_by="account_id",
            )
            .agg(
                [
                    pl.col(
                        "incoming_amount_at_timestamp"
                    )
                    .sum()
                    .alias(
                        f"account_incoming_amount_{suffix}"
                    ),

                    pl.col(
                        "outgoing_amount_at_timestamp"
                    )
                    .sum()
                    .alias(
                        f"account_outgoing_amount_{suffix}"
                    ),
                ]
            )
        )

        if result is None:
            result = rolling
        else:
            result = (
                result
                .join(
                    rolling,
                    on=[
                        "account_id",
                        "timestamp",
                    ],
                    how="left",
                )
            )

    assert result is not None

    return result


def attach_account_flow(
    base: pl.LazyFrame,
) -> pl.LazyFrame:

    flow = (
        build_rolling_flow_table(
            base
        )
    )

    sender_flow = (
        flow
        .rename(
            {
                "account_id":
                    "sender_id",

                **{
                    (
                        f"account_incoming_amount_{suffix}"
                    ):
                    (
                        f"sender_recent_incoming_amount_{suffix}"
                    )
                    for suffix
                    in FLOW_WINDOWS
                },

                **{
                    (
                        f"account_outgoing_amount_{suffix}"
                    ):
                    (
                        f"sender_recent_outgoing_amount_{suffix}"
                    )
                    for suffix
                    in FLOW_WINDOWS
                },
            }
        )
    )

    receiver_flow = (
        flow
        .rename(
            {
                "account_id":
                    "receiver_id",

                **{
                    (
                        f"account_incoming_amount_{suffix}"
                    ):
                    (
                        f"receiver_recent_incoming_amount_{suffix}"
                    )
                    for suffix
                    in FLOW_WINDOWS
                },

                **{
                    (
                        f"account_outgoing_amount_{suffix}"
                    ):
                    (
                        f"receiver_recent_outgoing_amount_{suffix}"
                    )
                    for suffix
                    in FLOW_WINDOWS
                },
            }
        )
    )

    frame = (
        base
        .join(
            sender_flow,
            on=[
                "sender_id",
                "timestamp",
            ],
            how="left",
        )
        .join(
            receiver_flow,
            on=[
                "receiver_id",
                "timestamp",
            ],
            how="left",
        )
    )

    flow_columns = []

    for suffix in FLOW_WINDOWS:

        flow_columns.extend(
            [
                f"sender_recent_incoming_amount_{suffix}",
                f"sender_recent_outgoing_amount_{suffix}",

                f"receiver_recent_incoming_amount_{suffix}",
                f"receiver_recent_outgoing_amount_{suffix}",
            ]
        )

    return (
        frame
        .with_columns(
            [
                pl.col(column)
                .fill_null(0.0)
                for column
                in flow_columns
            ]
        )
    )


def add_sender_amount_deviation(
    frame: pl.LazyFrame,
) -> pl.LazyFrame:
    """
    Compare the current outgoing transaction amount with the
    sender's strictly-prior 24-hour behaviour.
    """

    frame = (
        frame
        .with_columns(
            [
                pl.when(
                    pl.col(
                        "sender_amount_count_24h"
                    )
                    > 0
                )
                .then(
                    pl.col(
                        "sender_amount_sum_24h"
                    )
                    /
                    pl.col(
                        "sender_amount_count_24h"
                    )
                )
                .otherwise(
                    None
                )
                .alias(
                    "sender_amount_mean_24h"
                )
            ]
        )
    )

    # E[X^2] - E[X]^2
    frame = (
        frame
        .with_columns(
            [
                pl.when(
                    pl.col(
                        "sender_amount_count_24h"
                    )
                    > 1
                )
                .then(
                    (
                        pl.col(
                            "sender_amount_sumsq_24h"
                        )
                        /
                        pl.col(
                            "sender_amount_count_24h"
                        )
                    )
                    -
                    (
                        pl.col(
                            "sender_amount_mean_24h"
                        )
                        *
                        pl.col(
                            "sender_amount_mean_24h"
                        )
                    )
                )
                .otherwise(
                    None
                )
                .alias(
                    "sender_amount_variance_24h"
                )
            ]
        )
    )

    # Floating-point roundoff can make variance slightly < 0.
    frame = (
        frame
        .with_columns(
            [
                pl.when(
                    pl.col(
                        "sender_amount_variance_24h"
                    )
                    >= 0
                )
                .then(
                    pl.col(
                        "sender_amount_variance_24h"
                    )
                    .sqrt()
                )
                .otherwise(
                    0.0
                )
                .alias(
                    "sender_amount_std_24h"
                )
            ]
        )
    )

    frame = (
        frame
        .with_columns(
            [
                pl.when(
                    pl.col(
                        "sender_amount_mean_24h"
                    )
                    > 0
                )
                .then(
                    pl.col(
                        "amount_paid"
                    )
                    /
                    pl.col(
                        "sender_amount_mean_24h"
                    )
                )
                .otherwise(
                    None
                )
                .alias(
                    "sender_amount_ratio_to_mean_24h"
                ),

                pl.when(
                    pl.col(
                        "sender_amount_std_24h"
                    )
                    > 0
                )
                .then(
                    (
                        pl.col(
                            "amount_paid"
                        )
                        -
                        pl.col(
                            "sender_amount_mean_24h"
                        )
                    )
                    /
                    pl.col(
                        "sender_amount_std_24h"
                    )
                )
                .otherwise(
                    None
                )
                .alias(
                    "sender_amount_zscore_24h"
                ),
            ]
        )
    )

    return frame


def add_receiver_amount_deviation(
    frame: pl.LazyFrame,
) -> pl.LazyFrame:

    frame = (
        frame
        .with_columns(
            [
                pl.when(
                    pl.col(
                        "receiver_amount_count_24h"
                    )
                    > 0
                )
                .then(
                    pl.col(
                        "receiver_amount_sum_24h"
                    )
                    /
                    pl.col(
                        "receiver_amount_count_24h"
                    )
                )
                .otherwise(
                    None
                )
                .alias(
                    "receiver_amount_mean_24h"
                )
            ]
        )
    )

    frame = (
        frame
        .with_columns(
            [
                pl.when(
                    pl.col(
                        "receiver_amount_count_24h"
                    )
                    > 1
                )
                .then(
                    (
                        pl.col(
                            "receiver_amount_sumsq_24h"
                        )
                        /
                        pl.col(
                            "receiver_amount_count_24h"
                        )
                    )
                    -
                    (
                        pl.col(
                            "receiver_amount_mean_24h"
                        )
                        *
                        pl.col(
                            "receiver_amount_mean_24h"
                        )
                    )
                )
                .otherwise(
                    None
                )
                .alias(
                    "receiver_amount_variance_24h"
                )
            ]
        )
    )

    frame = (
        frame
        .with_columns(
            [
                pl.when(
                    pl.col(
                        "receiver_amount_variance_24h"
                    )
                    >= 0
                )
                .then(
                    pl.col(
                        "receiver_amount_variance_24h"
                    )
                    .sqrt()
                )
                .otherwise(
                    0.0
                )
                .alias(
                    "receiver_amount_std_24h"
                ),

                pl.when(
                    pl.col(
                        "receiver_amount_mean_24h"
                    )
                    > 0
                )
                .then(
                    pl.col(
                        "amount_received"
                    )
                    /
                    pl.col(
                        "receiver_amount_mean_24h"
                    )
                )
                .otherwise(
                    None
                )
                .alias(
                    "receiver_amount_ratio_to_mean_24h"
                ),
            ]
        )
    )

    frame = (
        frame
        .with_columns(
            [
                pl.when(
                    pl.col(
                        "receiver_amount_std_24h"
                    )
                    > 0
                )
                .then(
                    (
                        pl.col(
                            "amount_received"
                        )
                        -
                        pl.col(
                            "receiver_amount_mean_24h"
                        )
                    )
                    /
                    pl.col(
                        "receiver_amount_std_24h"
                    )
                )
                .otherwise(
                    None
                )
                .alias(
                    "receiver_amount_zscore_24h"
                )
            ]
        )
    )

    return frame


def add_flow_ratios(
    frame: pl.LazyFrame,
) -> pl.LazyFrame:

    expressions = []

    # Focus on 1h and 24h for derived flow ratios.
    for role in [
        "sender",
        "receiver",
    ]:

        for suffix in [
            "1h",
            "24h",
        ]:

            incoming = (
                pl.col(
                    f"{role}_recent_incoming_amount_{suffix}"
                )
            )

            outgoing = (
                pl.col(
                    f"{role}_recent_outgoing_amount_{suffix}"
                )
            )

            # ------------------------------------------------
            # Incoming / outgoing ratio
            # ------------------------------------------------

            expressions.append(
                (
                    incoming
                    /
                    (
                        outgoing
                        + EPSILON
                    )
                )
                .alias(
                    f"{role}_inflow_outflow_ratio_{suffix}"
                )
            )

            # ------------------------------------------------
            # Symmetric balance:
            #
            #  +1 = mostly incoming
            #  -1 = mostly outgoing
            #   0 = balanced
            # ------------------------------------------------

            expressions.append(
                (
                    (
                        incoming
                        -
                        outgoing
                    )
                    /
                    (
                        incoming
                        +
                        outgoing
                        +
                        EPSILON
                    )
                )
                .alias(
                    f"{role}_flow_balance_{suffix}"
                )
            )

            # ------------------------------------------------
            # Pass-through:
            #
            # Similar incoming/outgoing magnitude -> near 1.
            # ------------------------------------------------

            expressions.append(
                (
                    pl.min_horizontal(
                        incoming,
                        outgoing,
                    )
                    /
                    (
                        pl.max_horizontal(
                            incoming,
                            outgoing,
                        )
                        +
                        EPSILON
                    )
                )
                .alias(
                    f"{role}_pass_through_ratio_{suffix}"
                )
            )

            # ------------------------------------------------
            # Retention relative to incoming money.
            # Only meaningful when inflow exists.
            # ------------------------------------------------

            expressions.append(
                pl.when(
                    incoming
                    > 0
                )
                .then(
                    (
                        incoming
                        -
                        outgoing
                    )
                    /
                    (
                        incoming
                        +
                        EPSILON
                    )
                )
                .otherwise(
                    None
                )
                .alias(
                    f"{role}_fund_retention_{suffix}"
                )
            )

    return (
        frame
        .with_columns(
            expressions
        )
    )


def drop_internal_columns(
    frame: pl.LazyFrame,
) -> pl.LazyFrame:
    """
    Remove helper values required to calculate the final
    features but not needed as model inputs.
    """

    drop_columns = [
        "sender_amount_variance_24h",
        "receiver_amount_variance_24h",
    ]

    return frame.drop(
        drop_columns
    )


def stage3_flow_deviation() -> None:

    print_step(
        "PHASE 2B — Stage 3: Flow and behavioural deviation"
    )

    base = scan(
        STAGE2_FILE
    )

    frame = (
        attach_account_flow(
            base
        )
    )

    frame = (
        add_sender_amount_deviation(
            frame
        )
    )

    frame = (
        add_receiver_amount_deviation(
            frame
        )
    )

    frame = (
        add_flow_ratios(
            frame
        )
    )

    frame = (
        drop_internal_columns(
            frame
        )
    )

    frame = (
        frame
        .sort(
            [
                "timestamp",
                "transaction_id",
            ]
        )
    )

    resolve_schema(
        frame,
        "Stage 3",
    )

    sink(
        frame,
        STAGE3_FILE,
    )

    verify_base_invariants(
        STAGE3_FILE
    )

    # Final feature file is a copy of validated Stage 3.
    if FINAL_FILE.exists():
        FINAL_FILE.unlink()

    shutil.copy2(
        STAGE3_FILE,
        FINAL_FILE,
    )

    print(
        f"\nFinal Phase 2B file:\n"
        f"{FINAL_FILE}"
    )


# ============================================================
# Metadata
# ============================================================

def write_metadata() -> None:

    schema = (
        scan(
            FINAL_FILE
        )
        .collect_schema()
    )

    metadata = {
        "dataset":
            "IBM AML HI-Small",

        "phase":
            "2B",

        "rows":
            EXPECTED_ROWS,

        "columns":
            len(schema.names()),

        "history_rule":
            (
                "All rolling historical windows use "
                "closed='left'. A transaction at timestamp t "
                "therefore uses only observations before t. "
                "Same-timestamp events remain simultaneous."
            ),

        "windows": {
            "velocity":
                list(
                    COUNT_WINDOWS.keys()
                ),

            "volume":
                list(
                    VOLUME_WINDOWS.keys()
                ),

            "counterparty":
                list(
                    COUNTERPARTY_WINDOWS.keys()
                ),

            "flow":
                list(
                    FLOW_WINDOWS.keys()
                ),
        },

        "feature_groups": {
            "velocity": [
                "sender_tx_count_10m",
                "sender_tx_count_1h",
                "sender_tx_count_6h",
                "sender_tx_count_24h",
                "sender_tx_count_3d",

                "receiver_tx_count_10m",
                "receiver_tx_count_1h",
                "receiver_tx_count_6h",
                "receiver_tx_count_24h",
                "receiver_tx_count_3d",
            ],

            "volume": [
                "sender_amount_sum_1h",
                "sender_amount_sum_6h",
                "sender_amount_sum_24h",
                "sender_amount_sum_3d",

                "receiver_amount_sum_1h",
                "receiver_amount_sum_6h",
                "receiver_amount_sum_24h",
                "receiver_amount_sum_3d",
            ],

            "counterparty": [
                "sender_unique_receivers_1h",
                "sender_unique_receivers_24h",

                "receiver_unique_senders_1h",
                "receiver_unique_senders_24h",

                "sender_receiver_prior_tx_count",
                "is_new_counterparty",
            ],

            "behavioural_deviation": [
                "sender_amount_mean_24h",
                "sender_amount_std_24h",
                "sender_amount_ratio_to_mean_24h",
                "sender_amount_zscore_24h",

                "receiver_amount_mean_24h",
                "receiver_amount_std_24h",
                "receiver_amount_ratio_to_mean_24h",
                "receiver_amount_zscore_24h",
            ],

            "flow": [
                "sender_recent_incoming_amount_1h",
                "sender_recent_outgoing_amount_1h",
                "sender_recent_incoming_amount_6h",
                "sender_recent_outgoing_amount_6h",
                "sender_recent_incoming_amount_24h",
                "sender_recent_outgoing_amount_24h",

                "sender_inflow_outflow_ratio_1h",
                "sender_flow_balance_1h",
                "sender_pass_through_ratio_1h",
                "sender_fund_retention_1h",

                "sender_inflow_outflow_ratio_24h",
                "sender_flow_balance_24h",
                "sender_pass_through_ratio_24h",
                "sender_fund_retention_24h",
            ],
        },

        "important_notes": {
            "no_target_usage":
                (
                    "is_laundering is carried through as the "
                    "target but is never used to construct "
                    "features."
                ),

            "same_timestamp":
                (
                    "Transactions sharing an exact timestamp "
                    "cannot see each other in rolling history."
                ),

            "real_time_compatibility":
                (
                    "Phase 2B features were selected so they "
                    "can later be recreated using an incremental "
                    "account-state store in the realtime scoring "
                    "service."
                ),
        },
    }

    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
        )


# ============================================================
# Orchestration
# ============================================================

def run_all() -> None:

    validate_inputs()
    ensure_directories()

    stage1_velocity_volume()
    stage2_counterparties()
    stage3_flow_deviation()

    write_metadata()

    print_step(
        "PHASE 2B COMPLETE"
    )

    print(
        f"Final features:\n"
        f"{FINAL_FILE}"
    )


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "IBM AML HI-Small Phase 2B "
            "behavioural feature pipeline"
        )
    )

    parser.add_argument(
        "--stage",
        choices=[
            "all",
            "1",
            "2",
            "3",
        ],
        default="all",
        help=(
            "Run all stages or restart from an "
            "individual stage."
        ),
    )

    args = parser.parse_args()

    validate_inputs()
    ensure_directories()

    if args.stage == "all":
        run_all()

    elif args.stage == "1":
        stage1_velocity_volume()

    elif args.stage == "2":

        if not STAGE1_FILE.exists():
            raise FileNotFoundError(
                "Stage 1 must exist before Stage 2."
            )

        stage2_counterparties()

    elif args.stage == "3":

        if not STAGE2_FILE.exists():
            raise FileNotFoundError(
                "Stage 2 must exist before Stage 3."
            )

        stage3_flow_deviation()
        write_metadata()


if __name__ == "__main__":
    main()