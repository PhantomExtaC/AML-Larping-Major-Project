from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ibm_aml"
    / "hi_small"
    / "HI-Small_accounts.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "hi_small"
)

OUTPUT_FILE = OUTPUT_DIR / "accounts.parquet"

REPORT_FILE = (
    PROJECT_ROOT
    / "results"
    / "reports"
    / "hi_small_accounts_profile.json"
)


EXPECTED_COLUMNS = [
    "Bank Name",
    "Bank ID",
    "Account Number",
    "Entity ID",
    "Entity Name",
]


# ============================================================
# Helpers
# ============================================================

def infer_entity_type(entity_name: str) -> str:
    """
    Extract the synthetic entity type from labels such as:

        Corporation #33520
        Partnership #35397
        Sole Proprietorship #50438
    """

    if pd.isna(entity_name):
        return "Unknown"

    value = str(entity_name).strip()

    if " #" in value:
        return value.rsplit(" #", 1)[0]

    return value


# ============================================================
# Main preprocessing
# ============================================================

def main() -> None:

    print("\nIBM AML HI-Small Account Preprocessing\n")

    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"Account file not found:\n{RAW_FILE}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(
        RAW_FILE,
        dtype={
            "Bank ID": "string",
            "Account Number": "string",
            "Entity ID": "string",
            "Entity Name": "string",
            "Bank Name": "string",
        },
    )

    if list(df.columns) != EXPECTED_COLUMNS:
        raise ValueError(
            "Unexpected account schema.\n\n"
            f"Expected:\n{EXPECTED_COLUMNS}\n\n"
            f"Received:\n{list(df.columns)}"
        )

    print(f"Rows loaded: {len(df):,}")

    # --------------------------------------------------------
    # Rename
    # --------------------------------------------------------

    df = df.rename(
        columns={
            "Bank Name": "bank_name",
            "Bank ID": "bank_id",
            "Account Number": "account_number",
            "Entity ID": "source_entity_id",
            "Entity Name": "entity_name",
        }
    )

    # --------------------------------------------------------
    # Normalize identifiers
    # --------------------------------------------------------

    df["bank_id"] = (
        df["bank_id"]
        .astype("string")
        .str.strip()
    )

    df["account_number"] = (
        df["account_number"]
        .astype("string")
        .str.strip()
    )

    df["source_entity_id"] = (
        df["source_entity_id"]
        .astype("string")
        .str.strip()
    )

    # --------------------------------------------------------
    # Canonical account ID
    # --------------------------------------------------------

    df["account_id"] = (
        df["bank_id"]
        + "::"
        + df["account_number"]
    )

    # --------------------------------------------------------
    # Entity type
    # --------------------------------------------------------

    df["entity_type"] = (
        df["entity_name"]
        .apply(infer_entity_type)
        .astype("string")
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if df["account_id"].isna().any():
        raise ValueError(
            "Null account_id detected."
        )

    duplicate_accounts = int(
        df["account_id"].duplicated().sum()
    )

    missing_values = {
        col: int(count)
        for col, count
        in df.isna().sum().items()
    }

    entity_type_counts = {
        str(key): int(value)
        for key, value
        in df["entity_type"]
        .value_counts()
        .items()
    }

    # --------------------------------------------------------
    # Final layout
    # --------------------------------------------------------

    df = df[
        [
            "account_id",
            "bank_id",
            "account_number",
            "bank_name",
            "source_entity_id",
            "entity_name",
            "entity_type",
        ]
    ]

    # --------------------------------------------------------
    # Write Parquet
    # --------------------------------------------------------

    table = pa.Table.from_pandas(
        df,
        preserve_index=False,
    )

    pq.write_table(
        table,
        OUTPUT_FILE,
        compression="zstd",
        use_dictionary=True,
    )

    # --------------------------------------------------------
    # Profile
    # --------------------------------------------------------

    profile = {
        "rows": len(df),

        "unique_accounts": int(
            df["account_id"].nunique()
        ),

        "unique_banks": int(
            df["bank_id"].nunique()
        ),

        "unique_source_entities": int(
            df["source_entity_id"].nunique()
        ),

        "duplicate_account_ids": (
            duplicate_accounts
        ),

        "missing_values": missing_values,

        "entity_type_distribution": (
            entity_type_counts
        ),
    }

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            profile,
            file,
            indent=2,
        )

    print("\nAccount preprocessing complete.")

    print(
        f"Accounts: "
        f"{profile['rows']:,}"
    )

    print(
        f"Unique account IDs: "
        f"{profile['unique_accounts']:,}"
    )

    print(
        f"Unique banks: "
        f"{profile['unique_banks']:,}"
    )

    print(
        f"Duplicate IDs: "
        f"{profile['duplicate_account_ids']:,}"
    )

    print(
        f"\nOutput:\n{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()