from __future__ import annotations

import json
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

RESULTS_ROOT = (
    PROJECT_ROOT
    / "results"
)

MODEL_RESULTS_ROOT = (
    RESULTS_ROOT
    / "models"
)

COMPARISON_DIR = (
    RESULTS_ROOT
    / "comparisons"
    / "threshold_analysis"
)

COMPARISON_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Model prediction sources
# ============================================================

MODEL_SOURCES = {
    "xgboost_f4_v1": {
        "path":
            MODEL_RESULTS_ROOT
            / "xgboost_f4_v1"
            / "validation_predictions.parquet",

        "probability_column":
            "xgboost_probability",
    },

    "lightgbm_f4_v1": {
        "path":
            MODEL_RESULTS_ROOT
            / "lightgbm_f4_v1"
            / "validation_predictions.parquet",

        "probability_column":
            "lightgbm_probability",
    },

    # Add GraphSAGE here once the prediction parquet
    # has been copied from Drive to this local path.
    "graphsage_v1": {
        "path":
            MODEL_RESULTS_ROOT
            / "graphsage_v1"
            / "validation_predictions.parquet",

        "probability_column":
            "graphsage_v1_probability",
    },
}


# ============================================================
# Evaluation configuration
# ============================================================

RECALL_TARGETS = [
    0.80,
    0.90,
    0.95,
    0.99,
]

TOP_K_VALUES = [
    100,
    500,
    1000,
    5000,
    10000,
]


# ============================================================
# Utility functions
# ============================================================

def print_section(
    message: str,
) -> None:

    print(
        "\n"
        + "=" * 78
    )

    print(
        message
    )

    print(
        "=" * 78
    )


def safe_divide(
    numerator: np.ndarray | float,
    denominator: np.ndarray | float,
) -> np.ndarray | float:

    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(
            np.asarray(
                numerator,
                dtype=np.float64,
            ),
            dtype=np.float64,
        ),
        where=(
            np.asarray(
                denominator
            )
            != 0
        ),
    )


# ============================================================
# Prediction loading
# ============================================================

def load_predictions(
    model_name: str,
    config: dict,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:

    path = (
        config[
            "path"
        ]
    )

    probability_column = (
        config[
            "probability_column"
        ]
    )

    if not path.exists():

        raise FileNotFoundError(
            f"\nPrediction file not found "
            f"for {model_name}:\n"
            f"{path}"
        )

    df = pl.read_parquet(
        path,
        columns=[
            "transaction_id",
            "true_label",
            probability_column,
        ],
    )

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

    labels = (
        df
        .get_column(
            "true_label"
        )
        .to_numpy()
        .astype(
            np.int8,
            copy=False,
        )
    )

    probabilities = (
        df
        .get_column(
            probability_column
        )
        .to_numpy()
        .astype(
            np.float64,
            copy=False,
        )
    )

    if np.isnan(
        probabilities
    ).any():

        raise ValueError(
            f"{model_name} contains NaN probabilities."
        )

    if np.isinf(
        probabilities
    ).any():

        raise ValueError(
            f"{model_name} contains infinite probabilities."
        )

    return (
        transaction_ids,
        labels,
        probabilities,
    )


# ============================================================
# Threshold table
# ============================================================

def build_threshold_table(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> pl.DataFrame:
    """
    Construct exact operating points.

    Scores are sorted descending.

    At threshold T, every transaction with:
        probability >= T
    becomes an alert.

    Only the last row for each unique probability is retained,
    ensuring all tied probabilities are treated identically.
    """

    order = np.argsort(
        -probabilities,
        kind="stable",
    )

    sorted_probabilities = (
        probabilities[
            order
        ]
    )

    sorted_labels = (
        labels[
            order
        ]
        .astype(
            np.int64
        )
    )

    total_rows = (
        len(
            sorted_labels
        )
    )

    total_positive = int(
        sorted_labels.sum()
    )

    total_negative = (
        total_rows
        -
        total_positive
    )

    # --------------------------------------------------------
    # Cumulative predictions as threshold moves downward
    # --------------------------------------------------------

    tp_all = np.cumsum(
        sorted_labels
    )

    predicted_positive_all = (
        np.arange(
            1,
            total_rows + 1,
            dtype=np.int64,
        )
    )

    fp_all = (
        predicted_positive_all
        -
        tp_all
    )

    # --------------------------------------------------------
    # Keep only LAST position of each equal score
    # --------------------------------------------------------

    is_last_of_score = np.empty(
        total_rows,
        dtype=bool,
    )

    if total_rows > 1:

        is_last_of_score[
            :-1
        ] = (
            sorted_probabilities[
                :-1
            ]
            !=
            sorted_probabilities[
                1:
            ]
        )

    is_last_of_score[
        -1
    ] = True

    indices = np.flatnonzero(
        is_last_of_score
    )

    thresholds = (
        sorted_probabilities[
            indices
        ]
    )

    tp = (
        tp_all[
            indices
        ]
    )

    predicted_positive = (
        predicted_positive_all[
            indices
        ]
    )

    fp = (
        fp_all[
            indices
        ]
    )

    fn = (
        total_positive
        -
        tp
    )

    tn = (
        total_negative
        -
        fp
    )

    # --------------------------------------------------------
    # Derived metrics
    # --------------------------------------------------------

    precision = safe_divide(
        tp,
        predicted_positive,
    )

    recall = safe_divide(
        tp,
        total_positive,
    )

    f1 = safe_divide(
        2
        *
        precision
        *
        recall,

        precision
        +
        recall,
    )

    fpr = safe_divide(
        fp,
        total_negative,
    )

    fnr = safe_divide(
        fn,
        total_positive,
    )

    specificity = safe_divide(
        tn,
        total_negative,
    )

    alerts_per_10000 = (
        predicted_positive
        /
        total_rows
        *
        10000
    )

    false_positives_per_tp = safe_divide(
        fp,
        tp,
    )

    return pl.DataFrame(
        {
            "threshold":
                thresholds,

            "predicted_positive":
                predicted_positive,

            "tp":
                tp,

            "fp":
                fp,

            "tn":
                tn,

            "fn":
                fn,

            "precision":
                precision,

            "recall":
                recall,

            "f1":
                f1,

            "fpr":
                fpr,

            "fnr":
                fnr,

            "specificity":
                specificity,

            "alerts_per_10000":
                alerts_per_10000,

            "false_positives_per_tp":
                false_positives_per_tp,
        }
    )


# ============================================================
# Recall target selection
# ============================================================

def select_recall_targets(
    threshold_table: pl.DataFrame,
    model_name: str,
) -> pl.DataFrame:

    results = []

    for target in (
        RECALL_TARGETS
    ):

        eligible = (
            threshold_table
            .filter(
                pl.col(
                    "recall"
                )
                >= target
            )
        )

        if eligible.height == 0:
            continue

        # ----------------------------------------------------
        # Among thresholds satisfying recall target:
        # choose smallest alert burden.
        #
        # Equivalent to highest practical threshold.
        # ----------------------------------------------------

        best = (
            eligible
            .sort(
                [
                    "predicted_positive",
                    "fp",
                ]
            )
            .row(
                0,
                named=True,
            )
        )

        results.append(
            {
                "model":
                    model_name,

                "recall_target":
                    target,

                **best,
            }
        )

    return pl.DataFrame(
        results
    )


# ============================================================
# Best F1
# ============================================================

def select_best_f1(
    threshold_table: pl.DataFrame,
    model_name: str,
) -> dict:

    best = (
        threshold_table
        .sort(
            "f1",
            descending=True,
        )
        .row(
            0,
            named=True,
        )
    )

    return {
        "model":
            model_name,

        **best,
    }


# ============================================================
# Top-K metrics
# ============================================================

def top_k_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    model_name: str,
) -> pl.DataFrame:

    order = np.argsort(
        -probabilities,
        kind="stable",
    )

    sorted_labels = (
        labels[
            order
        ]
    )

    total_positive = (
        sorted_labels.sum()
    )

    results = []

    for requested_k in (
        TOP_K_VALUES
    ):

        k = min(
            requested_k,
            len(
                sorted_labels
            ),
        )

        top_labels = (
            sorted_labels[
                :k
            ]
        )

        tp = int(
            top_labels.sum()
        )

        precision = (
            tp
            /
            k
        )

        recall = (
            tp
            /
            total_positive
            if total_positive > 0
            else 0.0
        )

        results.append(
            {
                "model":
                    model_name,

                "k":
                    requested_k,

                "alerts":
                    k,

                "true_positives":
                    tp,

                "false_positives":
                    int(
                        k
                        -
                        tp
                    ),

                "precision_at_k":
                    float(
                        precision
                    ),

                "recall_at_k":
                    float(
                        recall
                    ),
            }
        )

    return pl.DataFrame(
        results
    )


# ============================================================
# General model summary
# ============================================================

def model_summary(
    model_name: str,
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> dict:

    positive_rate = float(
        labels.mean()
    )

    return {
        "model":
            model_name,

        "rows":
            int(
                len(
                    labels
                )
            ),

        "positives":
            int(
                labels.sum()
            ),

        "negatives":
            int(
                len(
                    labels
                )
                -
                labels.sum()
            ),

        "positive_rate":
            positive_rate,

        "random_pr_baseline":
            positive_rate,

        "pr_auc":
            float(
                average_precision_score(
                    labels,
                    probabilities,
                )
            ),

        "roc_auc":
            float(
                roc_auc_score(
                    labels,
                    probabilities,
                )
            ),
    }


# ============================================================
# Process one model
# ============================================================

def evaluate_model(
    model_name: str,
    config: dict,
) -> dict:

    print_section(
        f"Evaluating {model_name}"
    )

    (
        transaction_ids,
        labels,
        probabilities,
    ) = load_predictions(
        model_name,
        config,
    )

    summary = (
        model_summary(
            model_name,
            labels,
            probabilities,
        )
    )

    print(
        f"Rows:       "
        f"{summary['rows']:,}"
    )

    print(
        f"Positives:  "
        f"{summary['positives']:,}"
    )

    print(
        f"Base rate:  "
        f"{summary['positive_rate']:.6%}"
    )

    print(
        f"PR-AUC:     "
        f"{summary['pr_auc']:.6f}"
    )

    print(
        f"ROC-AUC:    "
        f"{summary['roc_auc']:.6f}"
    )

    # --------------------------------------------------------
    # Threshold scan
    # --------------------------------------------------------

    threshold_table = (
        build_threshold_table(
            labels,
            probabilities,
        )
    )

    threshold_file = (
        COMPARISON_DIR
        / f"{model_name}_threshold_metrics.parquet"
    )

    threshold_table.write_parquet(
        threshold_file,
        compression="zstd",
    )

    # --------------------------------------------------------
    # Recall targets
    # --------------------------------------------------------

    recall_targets = (
        select_recall_targets(
            threshold_table,
            model_name,
        )
    )

    recall_file = (
        COMPARISON_DIR
        / f"{model_name}_recall_targets.csv"
    )

    recall_targets.write_csv(
        recall_file
    )

    # --------------------------------------------------------
    # F1
    # --------------------------------------------------------

    best_f1 = (
        select_best_f1(
            threshold_table,
            model_name,
        )
    )

    # --------------------------------------------------------
    # Top-K
    # --------------------------------------------------------

    top_k = (
        top_k_metrics(
            labels,
            probabilities,
            model_name,
        )
    )

    top_k_file = (
        COMPARISON_DIR
        / f"{model_name}_top_k.csv"
    )

    top_k.write_csv(
        top_k_file
    )

    # --------------------------------------------------------
    # Console output
    # --------------------------------------------------------

    print(
        "\nRecall-constrained operating points:"
    )

    print(
        recall_targets.select(
            [
                "recall_target",
                "threshold",
                "precision",
                "recall",
                "tp",
                "fp",
                "fn",
                "alerts_per_10000",
                "false_positives_per_tp",
            ]
        )
    )

    print(
        "\nBest F1 operating point:"
    )

    print(
        {
            key:
                best_f1[key]

            for key in [
                "threshold",
                "precision",
                "recall",
                "f1",
                "tp",
                "fp",
                "fn",
                "alerts_per_10000",
            ]
        }
    )

    print(
        "\nTop-K performance:"
    )

    print(
        top_k
    )

    return {
        "summary":
            summary,

        "best_f1":
            best_f1,

        "recall_targets":
            recall_targets,

        "top_k":
            top_k,
    }


# ============================================================
# Main
# ============================================================

def main() -> None:

    print(
        "\n"
        "IBM AML HI-Small\n"
        "Phase 4A — Operational Threshold Analysis\n"
    )

    evaluated_models = []

    all_recall_targets = []
    all_top_k = []

    # --------------------------------------------------------
    # Evaluate every available model
    # --------------------------------------------------------

    for (
        model_name,
        config,
    ) in MODEL_SOURCES.items():

        path = (
            config[
                "path"
            ]
        )

        if not path.exists():

            print(
                f"\nSkipping {model_name}: "
                f"prediction file not found."
            )

            print(
                path
            )

            continue

        result = (
            evaluate_model(
                model_name,
                config,
            )
        )

        evaluated_models.append(
            {
                **result[
                    "summary"
                ],

                "best_f1":
                    float(
                        result[
                            "best_f1"
                        ][
                            "f1"
                        ]
                    ),

                "best_f1_threshold":
                    float(
                        result[
                            "best_f1"
                        ][
                            "threshold"
                        ]
                    ),

                "best_f1_precision":
                    float(
                        result[
                            "best_f1"
                        ][
                            "precision"
                        ]
                    ),

                "best_f1_recall":
                    float(
                        result[
                            "best_f1"
                        ][
                            "recall"
                        ]
                    ),
            }
        )

        all_recall_targets.append(
            result[
                "recall_targets"
            ]
        )

        all_top_k.append(
            result[
                "top_k"
            ]
        )

    # --------------------------------------------------------
    # Combined model summary
    # --------------------------------------------------------

    if len(
        evaluated_models
    ) == 0:

        raise RuntimeError(
            "No prediction files were found."
        )

    summary_df = (
        pl.DataFrame(
            evaluated_models
        )
        .sort(
            "pr_auc",
            descending=True,
        )
    )

    summary_file = (
        COMPARISON_DIR
        / "model_summary.csv"
    )

    summary_df.write_csv(
        summary_file
    )

    # --------------------------------------------------------
    # Combined recall targets
    # --------------------------------------------------------

    if all_recall_targets:

        recall_df = (
            pl.concat(
                all_recall_targets,
                how="vertical",
            )
            .sort(
                [
                    "recall_target",
                    "precision",
                ],
                descending=[
                    False,
                    True,
                ],
            )
        )

        recall_df.write_csv(
            COMPARISON_DIR
            / "recall_target_comparison.csv"
        )

    # --------------------------------------------------------
    # Combined Top-K
    # --------------------------------------------------------

    if all_top_k:

        top_k_df = (
            pl.concat(
                all_top_k,
                how="vertical",
            )
            .sort(
                [
                    "k",
                    "precision_at_k",
                ],
                descending=[
                    False,
                    True,
                ],
            )
        )

        top_k_df.write_csv(
            COMPARISON_DIR
            / "top_k_comparison.csv"
        )

    # --------------------------------------------------------
    # JSON operational summary
    # --------------------------------------------------------

    json_summary = {
        "evaluation_split":
            "validation",

        "models":
            evaluated_models,

        "recall_targets":
            RECALL_TARGETS,

        "top_k_values":
            TOP_K_VALUES,
    }

    with open(
        COMPARISON_DIR
        / "operational_summary.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            json_summary,
            file,
            indent=2,
        )

    # --------------------------------------------------------
    # Final display
    # --------------------------------------------------------

    print_section(
        "PHASE 4A COMPLETE"
    )

    print(
        "\nModel comparison:"
    )

    print(
        summary_df.select(
            [
                "model",
                "pr_auc",
                "roc_auc",
                "best_f1",
                "best_f1_precision",
                "best_f1_recall",
            ]
        )
    )

    print(
        "\nResults saved to:"
    )

    print(
        COMPARISON_DIR
    )


if __name__ == "__main__":
    main()