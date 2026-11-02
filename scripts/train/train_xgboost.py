from __future__ import annotations

import gc
import json
import os
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb

from tabular_common import (
    MODEL_DIR,
    RESULTS_DIR,
    TRAIN_FILE,
    VALIDATION_FILE,
    build_feature_definition,
    build_matrix,
    calculate_scale_pos_weight,
    print_step,
    probability_metrics,
    save_json,
    save_predictions,
    validate_files,
)


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "xgboost_f4_v1"

RUN_MODEL_DIR = (
    MODEL_DIR
    / MODEL_NAME
)

RUN_RESULTS_DIR = (
    RESULTS_DIR
    / MODEL_NAME
)

MODEL_FILE = (
    RUN_MODEL_DIR
    / "model.json"
)

FEATURE_DEFINITION_FILE = (
    RUN_MODEL_DIR
    / "feature_definition.json"
)

METRICS_FILE = (
    RUN_RESULTS_DIR
    / "validation_metrics.json"
)

PREDICTIONS_FILE = (
    RUN_RESULTS_DIR
    / "validation_predictions.parquet"
)

IMPORTANCE_FILE = (
    RUN_RESULTS_DIR
    / "feature_importance.parquet"
)


# ============================================================
# Main
# ============================================================

def main() -> None:

    print(
        "\n"
        "IBM AML HI-Small\n"
        "XGBoost F4 Baseline\n"
    )

    validate_files()

    RUN_MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RUN_RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Feature definition
    # --------------------------------------------------------

    print_step(
        "Building feature definition"
    )

    definition = (
        build_feature_definition()
    )

    feature_names = (
        definition[
            "feature_names"
        ]
    )

    print(
        f"Final feature count: "
        f"{len(feature_names)}"
    )

    save_json(
        definition,
        FEATURE_DEFINITION_FILE,
    )

    # --------------------------------------------------------
    # Training matrix
    # --------------------------------------------------------

    print_step(
        "Loading training matrix"
    )

    X_train, y_train, _ = (
        build_matrix(
            TRAIN_FILE,
            definition,
        )
    )

    print(
        "Train shape:",
        X_train.shape,
    )

    print(
        "Train positives:",
        int(y_train.sum()),
    )

    scale_pos_weight = (
        calculate_scale_pos_weight(
            y_train
        )
    )

    print(
        f"scale_pos_weight: "
        f"{scale_pos_weight:.4f}"
    )

    # --------------------------------------------------------
    # Validation matrix
    # --------------------------------------------------------

    print_step(
        "Loading validation matrix"
    )

    (
        X_validation,
        y_validation,
        _,
    ) = build_matrix(
        VALIDATION_FILE,
        definition,
    )

    print(
        "Validation shape:",
        X_validation.shape,
    )

    # --------------------------------------------------------
    # DMatrix
    # --------------------------------------------------------

    print_step(
        "Constructing XGBoost matrices"
    )

    dtrain = xgb.DMatrix(
        X_train,
        label=y_train,
        feature_names=feature_names,
        missing=np.nan,
        nthread=(
            os.cpu_count()
            or 1
        ),
    )

    dvalidation = xgb.DMatrix(
        X_validation,
        label=y_validation,
        feature_names=feature_names,
        missing=np.nan,
        nthread=(
            os.cpu_count()
            or 1
        ),
    )

    # DMatrix now owns its internal representation.
    del X_train
    del X_validation

    gc.collect()

    # --------------------------------------------------------
    # Parameters
    # --------------------------------------------------------

    params = {
        "objective":
            "binary:logistic",

        "eval_metric":
            "aucpr",

        "tree_method":
            "hist",

        "learning_rate":
            0.05,

        "max_depth":
            6,

        "min_child_weight":
            5,

        "subsample":
            0.80,

        "colsample_bytree":
            0.80,

        "alpha":
            0.10,

        "lambda":
            1.00,

        "scale_pos_weight":
            scale_pos_weight,

        "max_bin":
            256,

        "seed":
            42,

        "nthread":
            (
                os.cpu_count()
                or 1
            ),
    }

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print_step(
        "Training XGBoost"
    )

    booster = xgb.train(
        params=params,

        dtrain=dtrain,

        num_boost_round=3000,

        evals=[
            (
                dtrain,
                "train",
            ),

            (
                dvalidation,
                "validation",
            ),
        ],

        early_stopping_rounds=100,

        verbose_eval=25,
    )

    print(
        "\nBest iteration:",
        booster.best_iteration,
    )

    print(
        "Best validation PR-AUC:",
        booster.best_score,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    booster.save_model(
        str(MODEL_FILE)
    )

    # --------------------------------------------------------
    # Validation predictions from BEST iteration only
    # --------------------------------------------------------

    probabilities = booster.predict(
        dvalidation,
        iteration_range=(
            0,
            booster.best_iteration + 1,
        ),
    )

    metrics = probability_metrics(
        y_validation,
        probabilities,
    )

    metrics.update(
        {
            "model":
                MODEL_NAME,

            "best_iteration":
                int(
                    booster.best_iteration
                ),

            "xgboost_best_score":
                float(
                    booster.best_score
                ),

            "scale_pos_weight":
                float(
                    scale_pos_weight
                ),

            "feature_count":
                len(feature_names),
        }
    )

    print(
        "\nValidation metrics:"
    )

    print(
        json.dumps(
            metrics,
            indent=2,
        )
    )

    save_json(
        metrics,
        METRICS_FILE,
    )

    # We did not retain transaction IDs in memory.
    # Load only that tiny column now.
    transaction_ids = (
        pl.read_parquet(
            VALIDATION_FILE,
            columns=[
                "transaction_id"
            ],
        )
        .get_column(
            "transaction_id"
        )
        .to_numpy()
    )

    save_predictions(
        transaction_ids,
        y_validation,
        probabilities,
        PREDICTIONS_FILE,
        probability_column=(
            "xgboost_probability"
        ),
    )

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    gain_dictionary = (
        booster.get_score(
            importance_type="gain"
        )
    )

    weight_dictionary = (
        booster.get_score(
            importance_type="weight"
        )
    )

    gain_scores = []
    weight_scores = []

    for feature in feature_names:
        g = gain_dictionary.get(feature, 0.0)
        w = weight_dictionary.get(feature, 0.0)

        # Satisfy strict type checkers by handling the list[float] signature
        gain_scores.append(float(g[0] if isinstance(g, list) else g))
        weight_scores.append(float(w[0] if isinstance(w, list) else w))

    importance = pl.DataFrame(
        {
            "feature": feature_names,
            "gain": gain_scores,
            "weight": weight_scores,
        }
    ).sort(
        "gain",
        descending=True,
    )

    importance.write_parquet(
        IMPORTANCE_FILE,
        compression="zstd",
    )

    print_step(
        "XGBOOST COMPLETE"
    )

    print(
        f"Model:\n{MODEL_FILE}"
    )

    print(
        f"\nMetrics:\n{METRICS_FILE}"
    )

    print(
        f"\nPredictions:\n"
        f"{PREDICTIONS_FILE}"
    )


if __name__ == "__main__":
    main()