from __future__ import annotations

import gc
import json
import math
import re
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SPLIT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "hi_small"
    / "modeling"
    / "splits"
)

TRAIN_FILE = SPLIT_DIR / "train.parquet"
VALIDATION_FILE = SPLIT_DIR / "validation.parquet"
TEST_FILE = SPLIT_DIR / "test.parquet"
TAIL_FILE = SPLIT_DIR / "tail_stress.parquet"

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "models"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "research"
    / "models"
    / "baseline"
)


# ============================================================
# Columns
# ============================================================

TARGET = "is_laundering"

CATEGORICAL_COLUMNS = [
    "payment_format",
    "payment_currency",
    "receiving_currency",
]

NON_MODEL_COLUMNS = {
    TARGET,
    "transaction_id",
    "timestamp",

    "sender_id",
    "receiver_id",

    "from_bank",
    "to_bank",

    "from_account",
    "to_account",

    "sender_entity_type",
    "receiver_entity_type",
}

# These were useful intermediate statistics for calculating
# behavioural features but are not necessary as model inputs.
HELPER_PATTERNS = [
    "_amount_sumsq_",
    "_amount_count_",
]


# ============================================================
# General helpers
# ============================================================

def print_step(message: str) -> None:
    print("\n" + "=" * 78)
    print(message)
    print("=" * 78)


def validate_files() -> None:

    for path in [
        TRAIN_FILE,
        VALIDATION_FILE,
        TEST_FILE,
        TAIL_FILE,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing frozen split:\n{path}"
            )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def safe_name(value: str) -> str:
    """
    Convert category values into safe feature-name components.
    """

    value = str(value)

    value = re.sub(
        r"[^A-Za-z0-9]+",
        "_",
        value,
    )

    return value.strip("_").lower()


# ============================================================
# Feature discovery
# ============================================================

def get_numeric_features() -> list[str]:
    """
    Determine usable numerical features from the training schema.
    """

    schema = (
        pl.scan_parquet(
            str(TRAIN_FILE)
        )
        .collect_schema()
    )

    features = []

    for column in schema.names():

        if column in NON_MODEL_COLUMNS:
            continue

        if column in CATEGORICAL_COLUMNS:
            continue

        if any(
            pattern in column
            for pattern
            in HELPER_PATTERNS
        ):
            continue

        dtype = schema[column]

        # Everything remaining should be numeric.
        if dtype in {
            pl.Int8,
            pl.Int16,
            pl.Int32,
            pl.Int64,

            pl.UInt8,
            pl.UInt16,
            pl.UInt32,
            pl.UInt64,

            pl.Float32,
            pl.Float64,
        }:
            features.append(
                column
            )

    return features


def get_category_values() -> dict[str, list[str]]:
    """
    Learn categories from TRAIN ONLY.

    Validation/test categories unseen during training become
    all-zero one-hot vectors.
    """

    train = (
        pl.scan_parquet(
            str(TRAIN_FILE)
        )
    )

    categories = {}

    for column in CATEGORICAL_COLUMNS:

        values = (
            train
            .select(
                pl.col(column)
                .drop_nulls()
                .unique()
                .sort()
            )
            .collect()
            .get_column(column)
            .to_list()
        )

        categories[column] = [
            str(value)
            for value in values
        ]

    return categories


def build_feature_definition() -> dict:

    numeric_features = (
        get_numeric_features()
    )

    category_values = (
        get_category_values()
    )

    one_hot_features = []

    for column, values in (
        category_values.items()
    ):

        for value in values:

            one_hot_features.append(
                (
                    column,
                    value,
                    f"{column}__{safe_name(value)}",
                )
            )

    feature_names = (
        numeric_features
        +
        [
            output_name
            for _, _, output_name
            in one_hot_features
        ]
    )

    return {
        "numeric_features":
            numeric_features,

        "category_values":
            category_values,

        "one_hot_features":
            one_hot_features,

        "feature_names":
            feature_names,
    }


# ============================================================
# Matrix builder
# ============================================================

def build_matrix(
    path: Path,
    definition: dict,
    keep_transaction_ids: bool = False,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
]:
    """
    Materialize one split as a float32 NumPy matrix.

    This avoids pandas object overhead and keeps memory lower.
    """

    numeric_features = (
        definition[
            "numeric_features"
        ]
    )

    one_hot_features = (
        definition[
            "one_hot_features"
        ]
    )

    expressions: list[pl.Expr] = []

    # --------------------------------------------------------
    # Numeric features
    # --------------------------------------------------------

    for column in numeric_features:

        expressions.append(
            pl.col(column)
            .cast(pl.Float32)
            .alias(column)
        )

    # --------------------------------------------------------
    # One-hot categoricals
    # --------------------------------------------------------

    for (
        source_column,
        category,
        output_column,
    ) in one_hot_features:

        expressions.append(
            (
                pl.col(
                    source_column
                )
                == category
            )
            .fill_null(False)
            .cast(pl.Float32)
            .alias(
                output_column
            )
        )

    select_columns = [
        *expressions,

        pl.col(TARGET)
        .cast(pl.Int8),
    ]

    if keep_transaction_ids:
        select_columns.append(
            pl.col(
                "transaction_id"
            )
        )

    print(
        f"Loading: {path.name}"
    )

    df = (
        pl.scan_parquet(
            str(path)
        )
        .select(
            select_columns
        )
        .collect(
            engine="streaming"
        )
    )

    y = (
        df
        .get_column(TARGET)
        .to_numpy()
        .astype(
            np.int8,
            copy=False,
        )
    )

    transaction_ids = None

    if keep_transaction_ids:

        transaction_ids = (
            df
            .get_column(
                "transaction_id"
            )
            .to_numpy()
            .astype(
                np.int64,
                copy=False,
            )
        )

    X = (
        df
        .select(
            definition[
                "feature_names"
            ]
        )
        .to_numpy(
            order="c"
        )
        .astype(
            np.float32,
            copy=False,
        )
    )

    del df
    gc.collect()

    # --------------------------------------------------------
    # Convert +/- infinity to NaN.
    #
    # Tree models handle NaN natively.
    # Processing one column at a time avoids allocating one
    # giant boolean matrix.
    # --------------------------------------------------------

    for index in range(
        X.shape[1]
    ):
        column = X[:, index]

        infinite = np.isinf(
            column
        )

        if infinite.any():
            column[
                infinite
            ] = np.nan

    return (
        X,
        y,
        transaction_ids,
    )


# ============================================================
# Class imbalance
# ============================================================

def calculate_scale_pos_weight(
    y: np.ndarray,
) -> float:

    positives = int(
        y.sum()
    )

    negatives = int(
        len(y) - positives
    )

    if positives == 0:
        raise ValueError(
            "Training data has zero positive labels."
        )

    return (
        negatives
        / positives
    )


# ============================================================
# Evaluation
# ============================================================

def probability_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> dict:

    return {
        "pr_auc":
            float(
                average_precision_score(
                    y_true,
                    probabilities,
                )
            ),

        "roc_auc":
            float(
                roc_auc_score(
                    y_true,
                    probabilities,
                )
            ),

        "positives":
            int(
                y_true.sum()
            ),

        "rows":
            int(
                len(y_true)
            ),

        "positive_rate":
            float(
                y_true.mean()
            ),
    }


def save_predictions(
    transaction_ids: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
    output_file: Path,
    probability_column: str,
) -> None:

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pl.DataFrame(
        {
            "transaction_id":
                transaction_ids,

            "true_label":
                labels,

            probability_column:
                probabilities,
        }
    ).write_parquet(
        output_file,
        compression="zstd",
    )


def save_json(
    data: dict,
    path: Path,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            indent=2,
        )