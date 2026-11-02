from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

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

MODEL_NAME = "lightgbm_f4_v1"

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
    / "model.txt"
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
        "LightGBM F4 Baseline\n"
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
    # Feature contract
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
        f"Numeric features: "
        f"{len(definition['numeric_features'])}"
    )

    print(
        f"Final features: "
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
        # Model parameters
        # --------------------------------------------------------
    
    params = {
            "objective":
                "binary",
    
            "metric":
                "average_precision",
    
            "learning_rate":
                0.05,
    
            "num_leaves":
                63,
    
            "max_depth":
                -1,
    
            "min_data_in_leaf":
                100,
    
            "feature_fraction":
                0.80,
    
            "bagging_fraction":
                0.80,
    
            "bagging_freq":
                1,
    
            "lambda_l1":
                0.10,
    
            "lambda_l2":
                1.00,
    
            "scale_pos_weight":
                scale_pos_weight,
    
            "max_bin":
                255,
    
            "verbosity":
                -1,
    
            "seed":
                42,
    
            "feature_fraction_seed":
                42,
    
            "bagging_seed":
                42,
    
            "data_random_seed":
                42,
    
            "num_threads":
                -1,
    }

    # --------------------------------------------------------
    # LightGBM datasets
    # --------------------------------------------------------

    print_step(
        "Constructing LightGBM datasets"
    )

    train_set = lgb.Dataset(
        X_train,
        label=y_train,
        feature_name=feature_names,
        params=params,
        free_raw_data=True,
    )

    validation_set = lgb.Dataset(
        X_validation,
        label=y_validation,
        feature_name=feature_names,
        params=params,
        reference=train_set,
        free_raw_data=True,
    )

    # Force bin construction now.
    train_set.construct()
    validation_set.construct()

    # Raw arrays are no longer needed during training.
    del X_train
    del X_validation

    gc.collect()

    

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print_step(
        "Training LightGBM"
    )

    booster = lgb.train(
        params=params,

        train_set=train_set,

        num_boost_round=3000,

        valid_sets=[
            validation_set,
        ],

        valid_names=[
            "validation",
        ],

        callbacks=[
            lgb.early_stopping(
                stopping_rounds=100,
                first_metric_only=True,
                verbose=True,
            ),

            lgb.log_evaluation(
                period=25
            ),
        ],
    )

    print(
        "\nBest iteration:",
        booster.best_iteration,
    )

    print(
        "Best validation scores:",
        booster.best_score,
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    booster.save_model(
        str(MODEL_FILE),
        num_iteration=booster.best_iteration,
    )

    # Training Dataset objects can now be freed.
    del train_set
    del validation_set

    gc.collect()

    # --------------------------------------------------------
    # Reload validation only for final predictions
    # --------------------------------------------------------

    print_step(
        "Generating validation predictions"
    )

    (
        X_validation,
        y_validation,
        transaction_ids,
    ) = build_matrix(
        VALIDATION_FILE,
        definition,
        keep_transaction_ids=True,
    )

    probabilities = booster.predict(
        X_validation,
        num_iteration=booster.best_iteration,
    )

    # Satisfy strict type checkers by explicitly narrowing types
    assert isinstance(probabilities, np.ndarray)
    assert transaction_ids is not None

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

    save_predictions(
        transaction_ids,
        y_validation,
        probabilities,
        PREDICTIONS_FILE,
        probability_column=(
            "lightgbm_probability"
        ),
    )

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    importance = pl.DataFrame(
        {
            "feature":
                feature_names,

            "gain":
                booster.feature_importance(
                    importance_type="gain"
                ),

            "split":
                booster.feature_importance(
                    importance_type="split"
                ),
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
        "LIGHTGBM COMPLETE"
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