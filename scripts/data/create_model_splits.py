from __future__ import annotations

import json
from pathlib import Path

import polars as pl


# ============================================================
# Configuration
# ============================================================

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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "hi_small"
    / "modeling"
    / "splits"
)

TRAIN_FILE = OUTPUT_DIR / "train.parquet"
VALIDATION_FILE = OUTPUT_DIR / "validation.parquet"
TEST_FILE = OUTPUT_DIR / "test.parquet"
TAIL_FILE = OUTPUT_DIR / "tail_stress.parquet"

REPORT_FILE = (
    PROJECT_ROOT
    / "results"
    / "reports"
    / "split_profile.json"
)

DENSE_END = pl.datetime(
    2022, 9, 11, 0, 0, 0
)

TRAIN_TARGET = 0.70
VALIDATION_TARGET = 0.85

EXPECTED_ROWS = 5_078_345
EXPECTED_POSITIVES = 5_177


# ============================================================
# Helpers
# ============================================================

def print_step(text: str) -> None:
    print("\n" + "=" * 72)
    print(text)
    print("=" * 72)


def validate_input() -> None:

    if not FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"Phase 2B feature file missing:\n{FEATURE_FILE}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


def write_lazy(
    frame: pl.LazyFrame,
    path: Path,
) -> None:

    if path.exists():
        path.unlink()

    frame.sink_parquet(
        str(path),
        compression="zstd",
        statistics=True,
        maintain_order=True,
        engine="streaming",
    )


# ============================================================
# Determine chronological cutoffs
# ============================================================

def determine_cutoffs(
    dense: pl.LazyFrame,
) -> tuple:

    print_step(
        "Determining timestamp-safe chronological boundaries"
    )

    # Count rows at each exact timestamp.
    timestamp_counts = (
        dense
        .group_by("timestamp")
        .agg(
            pl.len().alias("rows")
        )
        .sort("timestamp")
        .collect()
    )

    total_rows = int(
        timestamp_counts["rows"].sum()
    )

    train_target_rows = (
        total_rows
        * TRAIN_TARGET
    )

    validation_target_rows = (
        total_rows
        * VALIDATION_TARGET
    )

    timestamp_counts = (
        timestamp_counts
        .with_columns(
            pl.col("rows")
            .cum_sum()
            .alias("cumulative_rows")
        )
    )

    train_cutoff = (
        timestamp_counts
        .filter(
            pl.col("cumulative_rows")
            >= train_target_rows
        )
        ["timestamp"][0]
    )

    validation_cutoff = (
        timestamp_counts
        .filter(
            pl.col("cumulative_rows")
            >= validation_target_rows
        )
        ["timestamp"][0]
    )

    print(
        f"Dense rows: {total_rows:,}"
    )

    print(
        f"Train cutoff:      {train_cutoff}"
    )

    print(
        f"Validation cutoff: {validation_cutoff}"
    )

    return (
        train_cutoff,
        validation_cutoff,
    )


# ============================================================
# Profile helper
# ============================================================

def profile_split(
    frame: pl.LazyFrame,
    name: str,
) -> dict:

    result = (
        frame
        .select(
            [
                pl.len()
                .alias("rows"),

                pl.col("is_laundering")
                .sum()
                .alias("positives"),

                pl.col("timestamp")
                .min()
                .alias("start"),

                pl.col("timestamp")
                .max()
                .alias("end"),
            ]
        )
        .collect()
        .to_dicts()[0]
    )

    rows = int(result["rows"])
    positives = int(
        result["positives"]
    )

    return {
        "name": name,

        "rows": rows,

        "positives": positives,

        "negatives": (
            rows - positives
        ),

        "positive_rate": (
            positives / rows
            if rows
            else 0
        ),

        "start": (
            str(result["start"])
            if result["start"]
            else None
        ),

        "end": (
            str(result["end"])
            if result["end"]
            else None
        ),
    }


# ============================================================
# Main
# ============================================================

def main() -> None:

    print(
        "\n"
        "IBM AML HI-Small\n"
        "Phase 2C — Chronological Split Generation\n"
    )

    validate_input()

    features = (
        pl.scan_parquet(
            str(FEATURE_FILE)
        )
    )

    # --------------------------------------------------------
    # Dense benchmark vs late stress/OOD tail
    # --------------------------------------------------------

    dense = (
        features
        .filter(
            pl.col("timestamp")
            < DENSE_END
        )
    )

    tail = (
        features
        .filter(
            pl.col("timestamp")
            >= DENSE_END
        )
    )

    # --------------------------------------------------------
    # Exact timestamp boundaries
    # --------------------------------------------------------

    (
        train_cutoff,
        validation_cutoff,
    ) = determine_cutoffs(
        dense
    )

    # No timestamp is divided between splits.
    train = (
        dense
        .filter(
            pl.col("timestamp")
            < train_cutoff
        )
    )

    validation = (
        dense
        .filter(
            (
                pl.col("timestamp")
                >= train_cutoff
            )
            &
            (
                pl.col("timestamp")
                < validation_cutoff
            )
        )
    )

    test = (
        dense
        .filter(
            pl.col("timestamp")
            >= validation_cutoff
        )
    )

    # --------------------------------------------------------
    # Profile BEFORE writing
    # --------------------------------------------------------

    print_step(
        "Profiling splits"
    )

    profiles = {
        "train":
            profile_split(
                train,
                "train",
            ),

        "validation":
            profile_split(
                validation,
                "validation",
            ),

        "test":
            profile_split(
                test,
                "test",
            ),

        "tail_stress":
            profile_split(
                tail,
                "tail_stress",
            ),
    }

    for name, profile in (
        profiles.items()
    ):

        print(
            f"\n{name.upper()}"
        )

        print(
            f"Rows:      "
            f"{profile['rows']:,}"
        )

        print(
            f"Positives: "
            f"{profile['positives']:,}"
        )

        print(
            f"Rate:      "
            f"{profile['positive_rate']:.6%}"
        )

        print(
            f"Period:    "
            f"{profile['start']} "
            f"→ {profile['end']}"
        )

    # --------------------------------------------------------
    # Sanity checks
    # --------------------------------------------------------

    benchmark_rows = (
        profiles["train"]["rows"]
        +
        profiles["validation"]["rows"]
        +
        profiles["test"]["rows"]
    )

    tail_rows = (
        profiles[
            "tail_stress"
        ]["rows"]
    )

    if (
        benchmark_rows
        + tail_rows
        != EXPECTED_ROWS
    ):
        raise ValueError(
            "Split row totals do not match "
            "the Phase 2B dataset."
        )

    benchmark_positives = (
        profiles["train"]["positives"]
        +
        profiles["validation"]["positives"]
        +
        profiles["test"]["positives"]
    )

    all_positives = (
        benchmark_positives
        +
        profiles[
            "tail_stress"
        ]["positives"]
    )

    if (
        all_positives
        != EXPECTED_POSITIVES
    ):
        raise ValueError(
            "Positive labels changed "
            "during splitting."
        )

    # Ensure each ordinary split is meaningful.
    for name in [
        "train",
        "validation",
        "test",
    ]:

        if (
            profiles[name]["positives"]
            < 100
        ):
            raise ValueError(
                f"{name} contains too few "
                "positive samples."
            )

    # --------------------------------------------------------
    # Write
    # --------------------------------------------------------

    print_step(
        "Writing frozen datasets"
    )

    write_lazy(
        train,
        TRAIN_FILE,
    )

    write_lazy(
        validation,
        VALIDATION_FILE,
    )

    write_lazy(
        test,
        TEST_FILE,
    )

    write_lazy(
        tail,
        TAIL_FILE,
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    report = {
        "dataset":
            "IBM AML HI-Small",

        "feature_source":
            str(FEATURE_FILE),

        "split_strategy":
            (
                "Chronological. Dense Sep 1-10 "
                "benchmark split by transaction "
                "count while never dividing an "
                "identical timestamp. Sep 11-18 "
                "preserved separately as a "
                "distribution-shift stress set."
            ),

        "train_target":
            TRAIN_TARGET,

        "validation_cumulative_target":
            VALIDATION_TARGET,

        "train_cutoff":
            str(train_cutoff),

        "validation_cutoff":
            str(validation_cutoff),

        "tail_start":
            "2022-09-11 00:00:00",

        "splits":
            profiles,
    }

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

    print_step(
        "PHASE 2C COMPLETE"
    )

    print(
        f"Frozen splits:\n"
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()