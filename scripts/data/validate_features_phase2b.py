from __future__ import annotations

import json
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

REPORT_FILE = (
    PROJECT_ROOT
    / "results"
    / "reports"
    / "phase2b_validation.json"
)

EXPECTED_ROWS = 5_078_345
EXPECTED_LAUNDERING = 5_177


REQUIRED_COLUMNS = [
    # Velocity
    "sender_tx_count_10m",
    "sender_tx_count_1h",
    "sender_tx_count_6h",
    "sender_tx_count_24h",

    "receiver_tx_count_10m",
    "receiver_tx_count_1h",
    "receiver_tx_count_6h",
    "receiver_tx_count_24h",

    # Volume
    "sender_amount_sum_1h",
    "sender_amount_sum_24h",
    "receiver_amount_sum_1h",
    "receiver_amount_sum_24h",

    # Counterparty
    "sender_unique_receivers_1h",
    "sender_unique_receivers_24h",

    "receiver_unique_senders_1h",
    "receiver_unique_senders_24h",

    "sender_receiver_prior_tx_count",
    "is_new_counterparty",

    # Deviation
    "sender_amount_mean_24h",
    "sender_amount_std_24h",
    "sender_amount_ratio_to_mean_24h",
    "sender_amount_zscore_24h",

    # Flow
    "sender_recent_incoming_amount_1h",
    "sender_recent_outgoing_amount_1h",

    "sender_flow_balance_1h",
    "sender_pass_through_ratio_1h",

    "sender_flow_balance_24h",
    "sender_pass_through_ratio_24h",
]


def main() -> None:

    print(
        "\n"
        "IBM AML HI-Small\n"
        "Phase 2B Validation\n"
    )

    if not FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"Feature file not found:\n"
            f"{FEATURE_FILE}"
        )

    lf = pl.scan_parquet(
        str(FEATURE_FILE)
    )

    schema = (
        lf.collect_schema()
    )

    missing_columns = (
        set(REQUIRED_COLUMNS)
        -
        set(schema.names())
    )

    if missing_columns:
        raise ValueError(
            "Missing Phase 2B features:\n"
            f"{sorted(missing_columns)}"
        )

    # ========================================================
    # Core integrity
    # ========================================================

    integrity = (
        lf
        .select(
            [
                pl.len()
                .alias(
                    "rows"
                ),

                pl.col(
                    "transaction_id"
                )
                .n_unique()
                .alias(
                    "unique_transaction_ids"
                ),

                pl.col(
                    "is_laundering"
                )
                .sum()
                .alias(
                    "laundering_rows"
                ),
            ]
        )
        .collect(
            engine="streaming"
        )
        .to_dicts()[0]
    )

    assert (
        integrity["rows"]
        == EXPECTED_ROWS
    )

    assert (
        integrity[
            "unique_transaction_ids"
        ]
        == EXPECTED_ROWS
    )

    assert (
        integrity[
            "laundering_rows"
        ]
        == EXPECTED_LAUNDERING
    )

    print(
        "Core integrity:",
        integrity,
    )

    # ========================================================
    # Negative-value checks
    # ========================================================

    count_columns = [
        column
        for column in REQUIRED_COLUMNS
        if (
            "_count_" in column
            or column.endswith(
                "_prior_tx_count"
            )
            or column.startswith(
                "sender_unique_"
            )
            or column.startswith(
                "receiver_unique_"
            )
        )
    ]

    negative_checks = (
        lf
        .select(
            [
                (
                    pl.col(column)
                    < 0
                )
                .sum()
                .alias(
                    column
                )
                for column
                in count_columns
            ]
        )
        .collect(
            engine="streaming"
        )
        .to_dicts()[0]
    )

    bad_negative_columns = {
        key: value
        for key, value
        in negative_checks.items()
        if value > 0
    }

    if bad_negative_columns:
        raise ValueError(
            "Negative historical counts detected:\n"
            f"{bad_negative_columns}"
        )

    # ========================================================
    # Rolling subset consistency
    #
    # 10m <= 1h <= 6h <= 24h
    # ========================================================

    rolling_violations = (
        lf
        .select(
            [
                (
                    pl.col(
                        "sender_tx_count_10m"
                    )
                    >
                    pl.col(
                        "sender_tx_count_1h"
                    )
                )
                .sum()
                .alias(
                    "sender_10m_gt_1h"
                ),

                (
                    pl.col(
                        "sender_tx_count_1h"
                    )
                    >
                    pl.col(
                        "sender_tx_count_6h"
                    )
                )
                .sum()
                .alias(
                    "sender_1h_gt_6h"
                ),

                (
                    pl.col(
                        "sender_tx_count_6h"
                    )
                    >
                    pl.col(
                        "sender_tx_count_24h"
                    )
                )
                .sum()
                .alias(
                    "sender_6h_gt_24h"
                ),

                (
                    pl.col(
                        "receiver_tx_count_10m"
                    )
                    >
                    pl.col(
                        "receiver_tx_count_1h"
                    )
                )
                .sum()
                .alias(
                    "receiver_10m_gt_1h"
                ),

                (
                    pl.col(
                        "receiver_tx_count_1h"
                    )
                    >
                    pl.col(
                        "receiver_tx_count_6h"
                    )
                )
                .sum()
                .alias(
                    "receiver_1h_gt_6h"
                ),

                (
                    pl.col(
                        "receiver_tx_count_6h"
                    )
                    >
                    pl.col(
                        "receiver_tx_count_24h"
                    )
                )
                .sum()
                .alias(
                    "receiver_6h_gt_24h"
                ),
            ]
        )
        .collect(
            engine="streaming"
        )
        .to_dicts()[0]
    )

    if any(
        value > 0
        for value
        in rolling_violations.values()
    ):
        raise ValueError(
            "Rolling-window containment failed:\n"
            f"{rolling_violations}"
        )

    # ========================================================
    # Unique counterparties cannot exceed transactions
    # ========================================================

    counterparty_violations = (
        lf
        .select(
            [
                (
                    pl.col(
                        "sender_unique_receivers_1h"
                    )
                    >
                    pl.col(
                        "sender_tx_count_1h"
                    )
                )
                .sum()
                .alias(
                    "sender_unique_gt_tx_1h"
                ),

                (
                    pl.col(
                        "sender_unique_receivers_24h"
                    )
                    >
                    pl.col(
                        "sender_tx_count_24h"
                    )
                )
                .sum()
                .alias(
                    "sender_unique_gt_tx_24h"
                ),

                (
                    pl.col(
                        "receiver_unique_senders_1h"
                    )
                    >
                    pl.col(
                        "receiver_tx_count_1h"
                    )
                )
                .sum()
                .alias(
                    "receiver_unique_gt_tx_1h"
                ),

                (
                    pl.col(
                        "receiver_unique_senders_24h"
                    )
                    >
                    pl.col(
                        "receiver_tx_count_24h"
                    )
                )
                .sum()
                .alias(
                    "receiver_unique_gt_tx_24h"
                ),
            ]
        )
        .collect(
            engine="streaming"
        )
        .to_dicts()[0]
    )

    if any(
        value > 0
        for value
        in counterparty_violations.values()
    ):
        raise ValueError(
            "Counterparty count consistency failed:\n"
            f"{counterparty_violations}"
        )

    # ========================================================
    # Relationship novelty consistency
    # ========================================================

    novelty_violations = (
        lf
        .filter(
            (
                pl.col(
                    "sender_receiver_prior_tx_count"
                )
                == 0
            )
            !=
            (
                pl.col(
                    "is_new_counterparty"
                )
                == 1
            )
        )
        .select(
            pl.len()
            .alias(
                "violations"
            )
        )
        .collect(
            engine="streaming"
        )
        .item()
    )

    if novelty_violations:
        raise ValueError(
            "is_new_counterparty does not match "
            "relationship history."
        )

    # ========================================================
    # Rolling count must never exceed lifetime prior count
    # ========================================================

    lifetime_violations = (
        lf
        .select(
            [
                (
                    pl.col(
                        "sender_tx_count_24h"
                    )
                    >
                    pl.col(
                        "sender_outgoing_prior_count"
                    )
                )
                .sum()
                .alias(
                    "sender"
                ),

                (
                    pl.col(
                        "receiver_tx_count_24h"
                    )
                    >
                    pl.col(
                        "receiver_incoming_prior_count"
                    )
                )
                .sum()
                .alias(
                    "receiver"
                ),
            ]
        )
        .collect(
            engine="streaming"
        )
        .to_dicts()[0]
    )

    if any(
        value > 0
        for value
        in lifetime_violations.values()
    ):
        raise ValueError(
            "Rolling history exceeds lifetime "
            "strictly-prior history."
        )

    # ========================================================
    # Flow range sanity
    # ========================================================

    flow_checks = (
        lf
        .select(
            [
                pl.col(
                    "sender_flow_balance_1h"
                )
                .min()
                .alias(
                    "flow_balance_min"
                ),

                pl.col(
                    "sender_flow_balance_1h"
                )
                .max()
                .alias(
                    "flow_balance_max"
                ),

                pl.col(
                    "sender_pass_through_ratio_1h"
                )
                .min()
                .alias(
                    "pass_through_min"
                ),

                pl.col(
                    "sender_pass_through_ratio_1h"
                )
                .max()
                .alias(
                    "pass_through_max"
                ),
            ]
        )
        .collect(
            engine="streaming"
        )
        .to_dicts()[0]
    )

    assert (
        flow_checks[
            "flow_balance_min"
        ]
        >= -1.000001
    )

    assert (
        flow_checks[
            "flow_balance_max"
        ]
        <= 1.000001
    )

    assert (
        flow_checks[
            "pass_through_min"
        ]
        >= 0
    )

    assert (
        flow_checks[
            "pass_through_max"
        ]
        <= 1.000001
    )

    # ========================================================
    # Null profile
    # ========================================================

    important_nullable = [
        "sender_amount_mean_24h",
        "sender_amount_std_24h",
        "sender_amount_ratio_to_mean_24h",
        "sender_amount_zscore_24h",

        "receiver_amount_mean_24h",
        "receiver_amount_std_24h",
        "receiver_amount_zscore_24h",
    ]

    null_profile = (
        lf
        .select(
            [
                pl.col(column)
                .is_null()
                .sum()
                .alias(
                    column
                )
                for column
                in important_nullable
            ]
        )
        .collect(
            engine="streaming"
        )
        .to_dicts()[0]
    )

    # Nulls here are valid:
    # no prior history => no historical mean/std/z-score.

    report = {
        "status":
            "PASS",

        "integrity":
            integrity,

        "rolling_window_violations":
            rolling_violations,

        "counterparty_violations":
            counterparty_violations,

        "lifetime_violations":
            lifetime_violations,

        "flow_checks":
            flow_checks,

        "nullable_feature_profile":
            null_profile,

        "columns":
            len(schema.names()),
    }

    REPORT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=2,
        )

    print(
        "\n"
        "Phase 2B validation PASSED."
    )

    print(
        f"\nReport:\n"
        f"{REPORT_FILE}"
    )


if __name__ == "__main__":
    main()