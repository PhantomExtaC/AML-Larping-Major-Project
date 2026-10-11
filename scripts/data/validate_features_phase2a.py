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
    / "phase2a"
    / "features.parquet"
)

REPORT_FILE = (
    PROJECT_ROOT
    / "results"
    / "reports"
    / "phase2a_validation.json"
)

EXPECTED_ROWS = 5_078_345
EXPECTED_LAUNDERING = 5_177


def main() -> None:

    print(
        "\nPhase 2A Feature Validation\n"
    )

    lf = pl.scan_parquet(
        FEATURE_FILE
    )

    # ========================================================
    # Basic integrity
    # ========================================================

    summary = (
        lf.select(
            [
                pl.len().alias("rows"),

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

                pl.col(
                    "sender_id"
                )
                .is_null()
                .sum()
                .alias(
                    "null_sender_ids"
                ),

                pl.col(
                    "receiver_id"
                )
                .is_null()
                .sum()
                .alias(
                    "null_receiver_ids"
                ),

                pl.col(
                    "sender_entity_type"
                )
                .is_null()
                .sum()
                .alias(
                    "null_sender_entity_type"
                ),

                pl.col(
                    "receiver_entity_type"
                )
                .is_null()
                .sum()
                .alias(
                    "null_receiver_entity_type"
                ),
            ]
        )
        .collect()
        .to_dicts()[0]
    )

    print(summary)

    # ========================================================
    # Hard assertions
    # ========================================================

    assert (
        summary["rows"]
        == EXPECTED_ROWS
    ), (
        "Row count changed during "
        "feature engineering."
    )

    assert (
        summary["unique_transaction_ids"]
        == EXPECTED_ROWS
    ), (
        "Duplicate or missing transaction IDs."
    )

    assert (
        summary["laundering_rows"]
        == EXPECTED_LAUNDERING
    ), (
        "Target labels changed."
    )

    assert (
        summary["null_sender_entity_type"]
        == 0
    ), (
        "Missing sender entity metadata."
    )

    assert (
        summary["null_receiver_entity_type"]
        == 0
    ), (
        "Missing receiver entity metadata."
    )

    # ========================================================
    # Temporal sanity
    # ========================================================

    temporal_summary = (
        lf.select(
            [
                pl.col(
                    "sender_observed_age_hours"
                )
                .min()
                .alias(
                    "min_sender_age"
                ),

                pl.col(
                    "receiver_observed_age_hours"
                )
                .min()
                .alias(
                    "min_receiver_age"
                ),

                pl.col(
                    "sender_prior_activity_count"
                )
                .min()
                .alias(
                    "min_sender_prior_count"
                ),

                pl.col(
                    "receiver_prior_activity_count"
                )
                .min()
                .alias(
                    "min_receiver_prior_count"
                ),

                pl.col(
                    "sender_seconds_since_previous_activity"
                )
                .min()
                .alias(
                    "min_sender_prev_seconds"
                ),

                pl.col(
                    "receiver_seconds_since_previous_activity"
                )
                .min()
                .alias(
                    "min_receiver_prev_seconds"
                ),
            ]
        )
        .collect()
        .to_dicts()[0]
    )

    print("\nTemporal summary:")
    print(temporal_summary)

    assert (
        temporal_summary[
            "min_sender_age"
        ]
        >= 0
    )

    assert (
        temporal_summary[
            "min_receiver_age"
        ]
        >= 0
    )

    assert (
        temporal_summary[
            "min_sender_prior_count"
        ]
        >= 0
    )

    assert (
        temporal_summary[
            "min_receiver_prior_count"
        ]
        >= 0
    )

    # Previous activity intervals,
    # when available, must be positive.
    if (
        temporal_summary[
            "min_sender_prev_seconds"
        ]
        is not None
    ):
        assert (
            temporal_summary[
                "min_sender_prev_seconds"
            ]
            > 0
        )

    if (
        temporal_summary[
            "min_receiver_prev_seconds"
        ]
        is not None
    ):
        assert (
            temporal_summary[
                "min_receiver_prev_seconds"
            ]
            > 0
        )

    # ========================================================
    # Save report
    # ========================================================

    report = {
        "integrity": summary,
        "temporal": temporal_summary,
        "status": "PASS",
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
        "\nPhase 2A validation PASSED."
    )

    print(
        f"\nReport:\n{REPORT_FILE}"
    )


if __name__ == "__main__":
    main()