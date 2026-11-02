import numpy as np
import pandas as pd

# Load the offline and online feature outputs
offline = pd.read_parquet(
    "data/processed/hi_small/modeling/phase2b/features.parquet"
)

online = pd.read_parquet(
    "data/processed/hi_small/online_replay/validation_features.parquet"
)

# Match records using transaction_id
assert offline["transaction_id"].is_unique
assert online["transaction_id"].is_unique

merged = offline.merge(
    online,
    on="transaction_id",
    suffixes=("_offline", "_online"),
    validate="one_to_one"
)

print("Matched transactions:", len(merged))

# Define feature groups using your actual schema/metadata.
# Replace these illustrative lists with the real feature names.
FEATURE_GROUPS = {
    "static_calendar": [],
    "account_history": [],
    "velocity_volume": [],
    "counterparty": [],
    "flow_deviation": []
}

# Exact comparison for counts, flags and categorical features.
EXACT_FEATURES = set([
    # Add integer counts, flags and categorical feature names here.
])

def calculate_group_metrics(feature_names):
    abs_diffs = []
    rel_diffs = []
    mismatches = 0

    for feature in feature_names:
        ref = merged[f"{feature}_offline"]
        pred = merged[f"{feature}_online"]

        if feature in EXACT_FEATURES:
            equal = (
                ref.eq(pred) |
                (ref.isna() & pred.isna())
            )
            mismatches += int((~equal).sum())

            if pd.api.types.is_numeric_dtype(ref):
                diff = (
                    ref.astype(float) - pred.astype(float)
                ).abs().to_numpy()

                abs_diffs.extend(diff[np.isfinite(diff)])

        else:
            ref_values = ref.to_numpy(dtype=float)
            pred_values = pred.to_numpy(dtype=float)

            valid = np.isfinite(ref_values) & np.isfinite(pred_values)

            ref_values = ref_values[valid]
            pred_values = pred_values[valid]

            diff = np.abs(ref_values - pred_values)

            equal = np.isclose(
                ref_values,
                pred_values,
                rtol=1e-5,
                atol=1e-5
            )

            mismatches += int((~equal).sum())
            abs_diffs.extend(diff)

            nonzero = np.abs(ref_values) > 1e-8
            if nonzero.any():
                rel = diff[nonzero] / np.abs(ref_values[nonzero])
                rel_diffs.extend(rel)

    return {
        "max_abs_diff": max(abs_diffs, default=0.0),
        "max_rel_diff": max(rel_diffs, default=0.0),
        "mismatches": mismatches
    }

for group, features in FEATURE_GROUPS.items():
    print(group, calculate_group_metrics(features))