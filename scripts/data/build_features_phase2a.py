from __future__ import annotations

import json
import math
from pathlib import Path

import polars as pl


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRANSACTION_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "hi_small"
    / "transactions"
)

ACCOUNTS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "hi_small"
    / "accounts.parquet"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "hi_small"
    / "modeling"
    / "phase2a"
)

OUTPUT_FILE = OUTPUT_DIR / "features.parquet"

SCHEMA_FILE = OUTPUT_DIR / "feature_schema.json"

METADATA_FILE = OUTPUT_DIR / "feature_metadata.json"


# ============================================================
# Helpers
# ============================================================

def print_step(message: str) -> None:
    print("\n" + "=" * 70)
    print(message)
    print("=" * 70)


def validate_inputs() -> None:
    """
    Ensure Phase 1 outputs exist before feature engineering begins.
    """

    if not TRANSACTION_DIR.exists():
        raise FileNotFoundError(
            f"Transaction directory not found:\n{TRANSACTION_DIR}"
        )

    transaction_files = list(
        TRANSACTION_DIR.glob("*.parquet")
    )

    if not transaction_files:
        raise FileNotFoundError(
            f"No transaction Parquet files found in:\n"
            f"{TRANSACTION_DIR}"
        )

    if not ACCOUNTS_FILE.exists():
        raise FileNotFoundError(
            f"Accounts file not found:\n{ACCOUNTS_FILE}"
        )

    print(
        f"Transaction Parquet parts found: "
        f"{len(transaction_files)}"
    )

    print(
        f"Accounts file found: "
        f"{ACCOUNTS_FILE}"
    )


# ============================================================
# Load canonical data
# ============================================================

def load_transactions() -> pl.LazyFrame:
    """
    Load all canonical transaction Parquet files lazily.
    """

    parquet_pattern = str(
        TRANSACTION_DIR / "*.parquet"
    )

    return pl.scan_parquet(
        parquet_pattern
    )


def load_accounts() -> pl.LazyFrame:
    """
    Load only account metadata required for Phase 2A.

    account_id
        joins directly to sender_id / receiver_id

    entity_type
        retained for later ablation experiments
    """

    return (
        pl.scan_parquet(
            ACCOUNTS_FILE
        )
        .select(
            [
                "account_id",
                "entity_type",
            ]
        )
    )


# ============================================================
# Static transaction features
# ============================================================

def add_static_features(
    transactions: pl.LazyFrame,
) -> pl.LazyFrame:
    """
    Add transaction-level features that require no historical
    information.

    These cannot cause temporal leakage.
    """

    return (
        transactions
        .with_columns(
            [
                # ------------------------------------------------
                # Cyclic hour representation
                # ------------------------------------------------

                (
                    (
                        2
                        * math.pi
                        * pl.col("hour").cast(pl.Float64)
                        / 24.0
                    ).sin()
                ).alias(
                    "hour_sin"
                ),

                (
                    (
                        2
                        * math.pi
                        * pl.col("hour").cast(pl.Float64)
                        / 24.0
                    ).cos()
                ).alias(
                    "hour_cos"
                ),

                # ------------------------------------------------
                # Cyclic day-of-week representation
                # ------------------------------------------------

                (
                    (
                        2
                        * math.pi
                        * pl.col("day_of_week").cast(pl.Float64)
                        / 7.0
                    ).sin()
                ).alias(
                    "day_of_week_sin"
                ),

                (
                    (
                        2
                        * math.pi
                        * pl.col("day_of_week").cast(pl.Float64)
                        / 7.0
                    ).cos()
                ).alias(
                    "day_of_week_cos"
                ),
            ]
        )
    )


# ============================================================
# Unified account activity table
# ============================================================

def build_account_activity_table(
    transactions: pl.LazyFrame,
) -> pl.LazyFrame:
    """
    Build a unified account activity stream.

    An account can appear as:

        sender
        receiver

    The goal is to answer questions like:

        How many transactions has this account participated in
        before the current timestamp?

    Same-timestamp events are treated as simultaneous.

    Self-transfers are deduplicated so:

        A -> A

    counts as ONE account activity, not two.
    """

    sender_events = (
        transactions
        .select(
            [
                pl.col(
                    "transaction_id"
                ),

                pl.col(
                    "timestamp"
                ),

                pl.col(
                    "sender_id"
                ).alias(
                    "account_id"
                ),
            ]
        )
    )

    receiver_events = (
        transactions
        .select(
            [
                pl.col(
                    "transaction_id"
                ),

                pl.col(
                    "timestamp"
                ),

                pl.col(
                    "receiver_id"
                ).alias(
                    "account_id"
                ),
            ]
        )
    )

    activity = pl.concat(
        [
            sender_events,
            receiver_events,
        ]
    )

    # --------------------------------------------------------
    # Remove duplicate self-transfer participation
    #
    # A -> A would otherwise appear once as sender and once
    # as receiver.
    # --------------------------------------------------------

    activity = (
        activity
        .unique(
            subset=[
                "account_id",
                "timestamp",
                "transaction_id",
            ]
        )
    )

    # --------------------------------------------------------
    # Collapse events occurring at identical timestamps
    #
    # Example:
    #
    # A  10:00  TX1
    # A  10:00  TX2
    # A  10:00  TX3
    #
    # becomes:
    #
    # A  10:00  3
    #
    # This lets all three transactions receive identical
    # historical features.
    # --------------------------------------------------------

    return (
        activity
        .group_by(
            [
                "account_id",
                "timestamp",
            ]
        )
        .agg(
            pl.len()
            .alias(
                "account_events_at_timestamp"
            )
        )
    )


# ============================================================
# Unified account history
# ============================================================

def add_account_history_features(
    account_activity: pl.LazyFrame,
) -> pl.LazyFrame:
    """
    Calculate leakage-safe historical account features.

    Historical features only represent timestamps STRICTLY
    before the current timestamp.

    Transactions at the exact same timestamp are simultaneous.
    """

    activity = (
        account_activity
        .sort(
            [
                "account_id",
                "timestamp",
            ]
        )
    )

    # --------------------------------------------------------
    # First time account appears anywhere in observed data
    # --------------------------------------------------------

    activity = (
        activity
        .with_columns(
            [
                pl.col(
                    "timestamp"
                )
                .min()
                .over(
                    "account_id"
                )
                .alias(
                    "account_first_seen"
                )
            ]
        )
    )

    # --------------------------------------------------------
    # Cumulative events INCLUDING current timestamp
    # --------------------------------------------------------

    activity = (
        activity
        .with_columns(
            [
                pl.col(
                    "account_events_at_timestamp"
                )
                .cum_sum()
                .over(
                    "account_id"
                )
                .alias(
                    "account_cumulative_events"
                )
            ]
        )
    )

    # --------------------------------------------------------
    # Strictly-prior activity count
    #
    # cumulative including current timestamp
    # minus events at current timestamp
    # --------------------------------------------------------

    activity = (
        activity
        .with_columns(
            [
                (
                    pl.col(
                        "account_cumulative_events"
                    )
                    -
                    pl.col(
                        "account_events_at_timestamp"
                    )
                )
                .cast(
                    pl.Int64
                )
                .alias(
                    "account_prior_activity_count"
                )
            ]
        )
    )

    # --------------------------------------------------------
    # Previous DISTINCT activity timestamp
    # --------------------------------------------------------

    activity = (
        activity
        .with_columns(
            [
                pl.col(
                    "timestamp"
                )
                .shift(
                    1
                )
                .over(
                    "account_id"
                )
                .alias(
                    "account_previous_activity_timestamp"
                )
            ]
        )
    )

    # --------------------------------------------------------
    # Observed tenure and time since previous activity
    # --------------------------------------------------------

    activity = (
        activity
        .with_columns(
            [
                (
                    pl.col(
                        "timestamp"
                    )
                    -
                    pl.col(
                        "account_first_seen"
                    )
                )
                .dt.total_seconds()
                .cast(
                    pl.Float64
                )
                .alias(
                    "account_observed_age_seconds"
                ),

                (
                    pl.col(
                        "timestamp"
                    )
                    -
                    pl.col(
                        "account_previous_activity_timestamp"
                    )
                )
                .dt.total_seconds()
                .cast(
                    pl.Float64
                )
                .alias(
                    "account_seconds_since_previous_activity"
                ),
            ]
        )
    )

    # --------------------------------------------------------
    # Human-friendly tenure units
    # --------------------------------------------------------

    activity = (
        activity
        .with_columns(
            [
                (
                    pl.col(
                        "account_observed_age_seconds"
                    )
                    / 3600.0
                )
                .alias(
                    "account_observed_age_hours"
                ),

                (
                    pl.col(
                        "account_observed_age_seconds"
                    )
                    / 86400.0
                )
                .alias(
                    "account_observed_age_days"
                ),
            ]
        )
    )

    # --------------------------------------------------------
    # Low-history / recently observed indicators
    # --------------------------------------------------------

    activity = (
        activity
        .with_columns(
            [
                (
                    pl.col(
                        "account_observed_age_hours"
                    )
                    < 1.0
                )
                .cast(
                    pl.Int8
                )
                .alias(
                    "account_is_new_1h"
                ),

                (
                    pl.col(
                        "account_observed_age_hours"
                    )
                    < 24.0
                )
                .cast(
                    pl.Int8
                )
                .alias(
                    "account_is_new_24h"
                ),

                (
                    pl.col(
                        "account_observed_age_hours"
                    )
                    < 72.0
                )
                .cast(
                    pl.Int8
                )
                .alias(
                    "account_is_new_3d"
                ),

                (
                    pl.col(
                        "account_prior_activity_count"
                    )
                    == 0
                )
                .cast(
                    pl.Int8
                )
                .alias(
                    "account_has_no_prior_activity"
                ),
            ]
        )
    )

    return activity


# ============================================================
# Direction-specific history
# ============================================================

def build_direction_history(
    transactions: pl.LazyFrame,
    account_column: str,
    prefix: str,
) -> pl.LazyFrame:
    """
    Build role-specific historical features.

    Sender example:

        sender_outgoing_prior_count

    Receiver example:

        receiver_incoming_prior_count

    Same-timestamp events are treated as simultaneous.
    """

    activity = (
        transactions
        .select(
            [
                pl.col(
                    account_column
                )
                .alias(
                    "account_id"
                ),

                pl.col(
                    "timestamp"
                ),

                pl.col(
                    "transaction_id"
                ),
            ]
        )
        .group_by(
            [
                "account_id",
                "timestamp",
            ]
        )
        .agg(
            pl.len()
            .alias(
                f"{prefix}_events_at_timestamp"
            )
        )
        .sort(
            [
                "account_id",
                "timestamp",
            ]
        )
    )

    # --------------------------------------------------------
    # Cumulative directional activity
    # --------------------------------------------------------

    activity = (
        activity
        .with_columns(
            [
                pl.col(
                    f"{prefix}_events_at_timestamp"
                )
                .cum_sum()
                .over(
                    "account_id"
                )
                .alias(
                    f"{prefix}_cumulative_events"
                ),

                pl.col(
                    "timestamp"
                )
                .shift(
                    1
                )
                .over(
                    "account_id"
                )
                .alias(
                    f"{prefix}_previous_timestamp"
                ),
            ]
        )
    )

    # --------------------------------------------------------
    # Strictly-prior directional activity
    # --------------------------------------------------------

    activity = (
        activity
        .with_columns(
            [
                (
                    pl.col(
                        f"{prefix}_cumulative_events"
                    )
                    -
                    pl.col(
                        f"{prefix}_events_at_timestamp"
                    )
                )
                .cast(
                    pl.Int64
                )
                .alias(
                    f"{prefix}_prior_count"
                ),

                (
                    pl.col(
                        "timestamp"
                    )
                    -
                    pl.col(
                        f"{prefix}_previous_timestamp"
                    )
                )
                .dt.total_seconds()
                .cast(
                    pl.Float64
                )
                .alias(
                    f"{prefix}_seconds_since_previous"
                ),
            ]
        )
    )

    return (
        activity
        .select(
            [
                "account_id",
                "timestamp",

                f"{prefix}_prior_count",

                f"{prefix}_seconds_since_previous",
            ]
        )
    )


# ============================================================
# Attach unified account history
# ============================================================

def attach_account_history(
    transactions: pl.LazyFrame,
    account_history: pl.LazyFrame,
) -> pl.LazyFrame:
    """
    Join unified account history to the transaction dataset
    for both sender and receiver.
    """

    sender_history = (
        account_history
        .select(
            [
                pl.col(
                    "account_id"
                )
                .alias(
                    "sender_id"
                ),

                pl.col(
                    "timestamp"
                ),

                pl.col(
                    "account_prior_activity_count"
                )
                .alias(
                    "sender_prior_activity_count"
                ),

                pl.col(
                    "account_observed_age_hours"
                )
                .alias(
                    "sender_observed_age_hours"
                ),

                pl.col(
                    "account_observed_age_days"
                )
                .alias(
                    "sender_observed_age_days"
                ),

                pl.col(
                    "account_seconds_since_previous_activity"
                )
                .alias(
                    "sender_seconds_since_previous_activity"
                ),

                pl.col(
                    "account_is_new_1h"
                )
                .alias(
                    "sender_is_new_1h"
                ),

                pl.col(
                    "account_is_new_24h"
                )
                .alias(
                    "sender_is_new_24h"
                ),

                pl.col(
                    "account_is_new_3d"
                )
                .alias(
                    "sender_is_new_3d"
                ),

                pl.col(
                    "account_has_no_prior_activity"
                )
                .alias(
                    "sender_has_no_prior_activity"
                ),
            ]
        )
    )

    receiver_history = (
        account_history
        .select(
            [
                pl.col(
                    "account_id"
                )
                .alias(
                    "receiver_id"
                ),

                pl.col(
                    "timestamp"
                ),

                pl.col(
                    "account_prior_activity_count"
                )
                .alias(
                    "receiver_prior_activity_count"
                ),

                pl.col(
                    "account_observed_age_hours"
                )
                .alias(
                    "receiver_observed_age_hours"
                ),

                pl.col(
                    "account_observed_age_days"
                )
                .alias(
                    "receiver_observed_age_days"
                ),

                pl.col(
                    "account_seconds_since_previous_activity"
                )
                .alias(
                    "receiver_seconds_since_previous_activity"
                ),

                pl.col(
                    "account_is_new_1h"
                )
                .alias(
                    "receiver_is_new_1h"
                ),

                pl.col(
                    "account_is_new_24h"
                )
                .alias(
                    "receiver_is_new_24h"
                ),

                pl.col(
                    "account_is_new_3d"
                )
                .alias(
                    "receiver_is_new_3d"
                ),

                pl.col(
                    "account_has_no_prior_activity"
                )
                .alias(
                    "receiver_has_no_prior_activity"
                ),
            ]
        )
    )

    return (
        transactions

        .join(
            sender_history,
            on=[
                "sender_id",
                "timestamp",
            ],
            how="left",
        )

        .join(
            receiver_history,
            on=[
                "receiver_id",
                "timestamp",
            ],
            how="left",
        )
    )


# ============================================================
# Attach direction-specific history
# ============================================================

def attach_direction_history(
    transactions: pl.LazyFrame,
    sender_history: pl.LazyFrame,
    receiver_history: pl.LazyFrame,
) -> pl.LazyFrame:
    """
    Attach:

        sender_outgoing_prior_count
        sender_outgoing_seconds_since_previous

        receiver_incoming_prior_count
        receiver_incoming_seconds_since_previous
    """

    sender_history = (
        sender_history
        .rename(
            {
                "account_id":
                    "sender_id"
            }
        )
    )

    receiver_history = (
        receiver_history
        .rename(
            {
                "account_id":
                    "receiver_id"
            }
        )
    )

    return (
        transactions

        .join(
            sender_history,
            on=[
                "sender_id",
                "timestamp",
            ],
            how="left",
        )

        .join(
            receiver_history,
            on=[
                "receiver_id",
                "timestamp",
            ],
            how="left",
        )
    )


# ============================================================
# Attach entity metadata
# ============================================================

def attach_entity_metadata(
    transactions: pl.LazyFrame,
    accounts: pl.LazyFrame,
) -> pl.LazyFrame:
    """
    Add sender/receiver entity type.

    These features will initially be excluded from the primary
    baseline and tested through a separate ablation experiment.
    """

    sender_accounts = (
        accounts
        .rename(
            {
                "account_id":
                    "sender_id",

                "entity_type":
                    "sender_entity_type",
            }
        )
    )

    receiver_accounts = (
        accounts
        .rename(
            {
                "account_id":
                    "receiver_id",

                "entity_type":
                    "receiver_entity_type",
            }
        )
    )

    return (
        transactions

        .join(
            sender_accounts,
            on="sender_id",
            how="left",
        )

        .join(
            receiver_accounts,
            on="receiver_id",
            how="left",
        )
    )


# ============================================================
# Feature schema validation
# ============================================================

def validate_feature_schema(
    features: pl.LazyFrame,
) -> None:
    """
    Resolve the lazy schema BEFORE running the expensive collect.

    This catches naming mistakes such as:

        sender_prior_outgoing_count

    vs

        sender_outgoing_prior_count

    before the 5-million-row computation begins.
    """

    required_columns = {
        "transaction_id",
        "timestamp",

        "sender_id",
        "receiver_id",

        "amount_paid",
        "amount_received",

        "hour_sin",
        "hour_cos",
        "day_of_week_sin",
        "day_of_week_cos",

        "sender_prior_activity_count",
        "receiver_prior_activity_count",

        "sender_observed_age_hours",
        "receiver_observed_age_hours",

        "sender_observed_age_days",
        "receiver_observed_age_days",

        "sender_seconds_since_previous_activity",
        "receiver_seconds_since_previous_activity",

        "sender_is_new_1h",
        "sender_is_new_24h",
        "sender_is_new_3d",

        "receiver_is_new_1h",
        "receiver_is_new_24h",
        "receiver_is_new_3d",

        "sender_has_no_prior_activity",
        "receiver_has_no_prior_activity",

        "sender_outgoing_prior_count",
        "sender_outgoing_seconds_since_previous",

        "receiver_incoming_prior_count",
        "receiver_incoming_seconds_since_previous",

        "sender_entity_type",
        "receiver_entity_type",

        "is_laundering",
    }

    schema = features.collect_schema()

    actual_columns = set(
        schema.names()
    )

    missing = (
        required_columns
        -
        actual_columns
    )

    if missing:
        raise ValueError(
            "\nPhase 2A schema validation failed.\n"
            f"Missing columns:\n"
            f"{sorted(missing)}"
        )

    print(
        f"Feature schema valid: "
        f"{len(actual_columns)} columns"
    )


# ============================================================
# Final cleanup
# ============================================================

def finalize_features(
    features: pl.LazyFrame,
) -> pl.LazyFrame:
    """
    Normalize safe null/count behaviour before output.

    Count features use 0 when no previous activity exists.

    Time-since-previous features remain NULL when there was no
    previous event because NULL and 0 have different meanings.
    """

    return (
        features

        .with_columns(
            [
                pl.col(
                    "sender_prior_activity_count"
                )
                .fill_null(
                    0
                )
                .cast(
                    pl.Int64
                ),

                pl.col(
                    "receiver_prior_activity_count"
                )
                .fill_null(
                    0
                )
                .cast(
                    pl.Int64
                ),

                pl.col(
                    "sender_outgoing_prior_count"
                )
                .fill_null(
                    0
                )
                .cast(
                    pl.Int64
                ),

                pl.col(
                    "receiver_incoming_prior_count"
                )
                .fill_null(
                    0
                )
                .cast(
                    pl.Int64
                ),
            ]
        )

        .sort(
            [
                "timestamp",
                "transaction_id",
            ]
        )
    )


# ============================================================
# Print resolved schema
# ============================================================

def print_feature_schema(
    features: pl.LazyFrame,
) -> None:

    print_step(
        "Checking final feature schema"
    )

    schema = (
        features
        .collect_schema()
    )

    for name, dtype in schema.items():
        print(
            f"{name:<55} {dtype}"
        )


# ============================================================
# Write dataset
# ============================================================

def write_features(
    features: pl.LazyFrame,
) -> pl.DataFrame:
    """
    Execute the lazy feature pipeline and write Phase 2A output.
    """

    print_step(
        "Collecting final Phase 2A feature dataset"
    )

    df = (
        features
        .collect(
            engine="streaming"
        )
    )

    print(
        f"Rows: "
        f"{df.height:,}"
    )

    print(
        f"Columns: "
        f"{df.width:,}"
    )

    print_step(
        "Writing Phase 2A Parquet"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.write_parquet(
        OUTPUT_FILE,
        compression="zstd",
        statistics=True,
    )

    return df


# ============================================================
# Write schema and metadata
# ============================================================

def write_metadata(
    df: pl.DataFrame,
) -> None:
    """
    Save machine-readable documentation describing the generated
    feature dataset.
    """

    schema = {
        name: str(dtype)
        for name, dtype
        in df.schema.items()
    }

    with open(
        SCHEMA_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            schema,
            file,
            indent=2,
        )

    metadata = {
        "dataset":
            "IBM AML HI-Small",

        "phase":
            "2A",

        "rows":
            df.height,

        "columns":
            df.width,

        "target":
            "is_laundering",

        "history_semantics": (
            "Historical features use only events whose timestamps "
            "are strictly earlier than the current transaction "
            "timestamp. Transactions sharing the same timestamp "
            "are treated as simultaneous."
        ),

        "observed_tenure_definition": (
            "Observed tenure is the elapsed time since the first "
            "transaction in which an account appears as either "
            "sender or receiver inside the available dataset. "
            "It is not the real bank-account creation age."
        ),

        "feature_groups": {
            "static": [
                "amount_paid",
                "amount_paid_log",

                "amount_received",
                "amount_received_log",

                "same_currency_amount_delta",

                "hour",
                "day_of_week",

                "hour_sin",
                "hour_cos",

                "day_of_week_sin",
                "day_of_week_cos",

                "is_cross_bank",
                "is_currency_conversion",
                "is_self_transfer",

                "payment_format",
                "payment_currency",
                "receiving_currency",
            ],

            "observed_tenure": [
                "sender_observed_age_hours",
                "sender_observed_age_days",

                "receiver_observed_age_hours",
                "receiver_observed_age_days",

                "sender_is_new_1h",
                "sender_is_new_24h",
                "sender_is_new_3d",

                "receiver_is_new_1h",
                "receiver_is_new_24h",
                "receiver_is_new_3d",
            ],

            "prior_activity": [
                "sender_prior_activity_count",
                "receiver_prior_activity_count",

                "sender_seconds_since_previous_activity",
                "receiver_seconds_since_previous_activity",
            ],

            "directional_history": [
                "sender_outgoing_prior_count",
                "sender_outgoing_seconds_since_previous",

                "receiver_incoming_prior_count",
                "receiver_incoming_seconds_since_previous",
            ],

            "entity_metadata": [
                "sender_entity_type",
                "receiver_entity_type",
            ],
        },

        "excluded_from_primary_model": [
            "transaction_id",

            "sender_id",
            "receiver_id",

            "from_bank",
            "to_bank",

            "from_account",
            "to_account",

            "sender_entity_type",
            "receiver_entity_type",
        ],

        "notes": {
            "entity_type": (
                "Entity type is retained for a separate ablation "
                "experiment because it originates from synthetic "
                "dataset metadata and may encode simulator-specific "
                "patterns."
            ),

            "time_since_previous": (
                "Time-since-previous fields remain null where no "
                "previous event exists. A null means no history; "
                "zero would incorrectly mean an event occurred "
                "zero seconds earlier."
            ),

            "same_timestamp": (
                "Multiple events involving an account at exactly "
                "the same timestamp cannot see each other as past "
                "history."
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
# Main
# ============================================================

def main() -> None:

    print(
        "\n"
        "IBM AML HI-Small\n"
        "Phase 2A Feature Engineering\n"
    )

    # --------------------------------------------------------
    # Validate Phase 1 outputs
    # --------------------------------------------------------

    validate_inputs()

    # --------------------------------------------------------
    # Load canonical data
    # --------------------------------------------------------

    print_step(
        "Loading canonical transactions"
    )

    transactions = (
        load_transactions()
    )

    accounts = (
        load_accounts()
    )

    # --------------------------------------------------------
    # Static features
    # --------------------------------------------------------

    print_step(
        "Building static features"
    )

    transactions = (
        add_static_features(
            transactions
        )
    )

    # --------------------------------------------------------
    # Unified sender + receiver account history
    # --------------------------------------------------------

    print_step(
        "Building unified account activity history"
    )

    account_activity = (
        build_account_activity_table(
            transactions
        )
    )

    account_history = (
        add_account_history_features(
            account_activity
        )
    )

    # --------------------------------------------------------
    # Sender-only directional history
    # --------------------------------------------------------

    print_step(
        "Building sender directional history"
    )

    sender_history = (
        build_direction_history(
            transactions,
            account_column="sender_id",
            prefix="sender_outgoing",
        )
    )

    # --------------------------------------------------------
    # Receiver-only directional history
    # --------------------------------------------------------

    print_step(
        "Building receiver directional history"
    )

    receiver_history = (
        build_direction_history(
            transactions,
            account_column="receiver_id",
            prefix="receiver_incoming",
        )
    )

    # --------------------------------------------------------
    # Attach unified account history
    # --------------------------------------------------------

    print_step(
        "Attaching account historical features"
    )

    features = (
        attach_account_history(
            transactions,
            account_history,
        )
    )

    # --------------------------------------------------------
    # Attach directional histories
    # --------------------------------------------------------

    print_step(
        "Attaching directional historical features"
    )

    features = (
        attach_direction_history(
            features,
            sender_history,
            receiver_history,
        )
    )

    # --------------------------------------------------------
    # Add entity metadata
    # --------------------------------------------------------

    print_step(
        "Attaching entity metadata"
    )

    features = (
        attach_entity_metadata(
            features,
            accounts,
        )
    )

    # --------------------------------------------------------
    # Validate names BEFORE expensive collect
    # --------------------------------------------------------

    print_step(
        "Validating Phase 2A feature contract"
    )

    validate_feature_schema(
        features
    )

    # --------------------------------------------------------
    # Final transformations
    # --------------------------------------------------------

    features = (
        finalize_features(
            features
        )
    )

    # --------------------------------------------------------
    # Final resolved schema
    # --------------------------------------------------------

    print_feature_schema(
        features
    )

    # --------------------------------------------------------
    # Execute + write
    # --------------------------------------------------------

    df = (
        write_features(
            features
        )
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    write_metadata(
        df
    )

    print_step(
        "Phase 2A complete"
    )

    print(
        f"Rows written: "
        f"{df.height:,}"
    )

    print(
        f"Columns written: "
        f"{df.width:,}"
    )

    print(
        f"\nFeature dataset:\n"
        f"{OUTPUT_FILE}"
    )

    print(
        f"\nSchema:\n"
        f"{SCHEMA_FILE}"
    )

    print(
        f"\nMetadata:\n"
        f"{METADATA_FILE}"
    )


if __name__ == "__main__":
    main()