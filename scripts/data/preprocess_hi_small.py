from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ibm_aml"
    / "hi_small"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "hi_small"
)

TRANSACTION_OUTPUT_DIR = OUTPUT_ROOT / "transactions"

SOURCE_FILE = RAW_DIR / "HI-Small_Trans.csv"

METADATA_FILE = OUTPUT_ROOT / "metadata.json"
SCHEMA_FILE = OUTPUT_ROOT / "schema.json"

CHUNK_SIZE = 500_000


# ============================================================
# Expected raw schema
# ============================================================

EXPECTED_COLUMNS = [
    "Timestamp",
    "From Bank",
    "Account",
    "To Bank",
    "Account.1",
    "Amount Received",
    "Receiving Currency",
    "Amount Paid",
    "Payment Currency",
    "Payment Format",
    "Is Laundering",
]


COLUMN_RENAME_MAP = {
    "Timestamp": "timestamp",
    "From Bank": "from_bank",
    "Account": "from_account",
    "To Bank": "to_bank",
    "Account.1": "to_account",
    "Amount Received": "amount_received",
    "Receiving Currency": "receiving_currency",
    "Amount Paid": "amount_paid",
    "Payment Currency": "payment_currency",
    "Payment Format": "payment_format",
    "Is Laundering": "is_laundering",
}


# ============================================================
# Setup
# ============================================================

def validate_source() -> None:
    if not SOURCE_FILE.exists():
        raise FileNotFoundError(
            f"Transaction file not found:\n{SOURCE_FILE}"
        )

    header = pd.read_csv(SOURCE_FILE, nrows=0)

    actual_columns = list(header.columns)

    if actual_columns != EXPECTED_COLUMNS:
        raise ValueError(
            "Unexpected source schema.\n\n"
            f"Expected:\n{EXPECTED_COLUMNS}\n\n"
            f"Received:\n{actual_columns}"
        )

    print("Source schema validated.")


def prepare_output_directory() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Start from a clean transaction dataset every run.
    if TRANSACTION_OUTPUT_DIR.exists():
        shutil.rmtree(TRANSACTION_OUTPUT_DIR)

    TRANSACTION_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# Transformations
# ============================================================

def normalize_chunk(
    chunk: pd.DataFrame,
    row_offset: int,
) -> pd.DataFrame:

    # --------------------------------------------------------
    # Rename source columns
    # --------------------------------------------------------

    chunk = chunk.rename(
        columns=COLUMN_RENAME_MAP
    ).copy()

    # --------------------------------------------------------
    # Stable transaction ID
    # --------------------------------------------------------

    chunk.insert(
        0,
        "transaction_id",
        np.arange(
            row_offset,
            row_offset + len(chunk),
            dtype=np.int64,
        ),
    )

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    chunk["timestamp"] = pd.to_datetime(
        chunk["timestamp"],
        errors="raise",
    )

    # --------------------------------------------------------
    # Preserve bank/account identifiers as strings
    # --------------------------------------------------------

    identifier_columns = [
        "from_bank",
        "from_account",
        "to_bank",
        "to_account",
    ]

    for column in identifier_columns:
        chunk[column] = chunk[column].astype(str)

    # --------------------------------------------------------
    # Stable account/entity identifiers
    # --------------------------------------------------------

    chunk["sender_id"] = (
        chunk["from_bank"]
        + "::"
        + chunk["from_account"]
    )

    chunk["receiver_id"] = (
        chunk["to_bank"]
        + "::"
        + chunk["to_account"]
    )

    # --------------------------------------------------------
    # Numerical types
    # --------------------------------------------------------

    chunk["amount_paid"] = pd.to_numeric(
        chunk["amount_paid"],
        errors="raise",
    ).astype("float64")

    chunk["amount_received"] = pd.to_numeric(
        chunk["amount_received"],
        errors="raise",
    ).astype("float64")

    chunk["is_laundering"] = (
        pd.to_numeric(
            chunk["is_laundering"],
            errors="raise",
        )
        .astype("int8")
    )

    # --------------------------------------------------------
    # Basic transaction flags
    # --------------------------------------------------------

    chunk["is_cross_bank"] = (
        chunk["from_bank"]
        != chunk["to_bank"]
    ).astype("int8")

    chunk["is_currency_conversion"] = (
        chunk["payment_currency"]
        != chunk["receiving_currency"]
    ).astype("int8")

    chunk["is_self_transfer"] = (
        chunk["sender_id"]
        == chunk["receiver_id"]
    ).astype("int8")

    # --------------------------------------------------------
    # Temporal fields
    # --------------------------------------------------------

    chunk["hour"] = (
        chunk["timestamp"]
        .dt.hour
        .astype("int8")
    )

    chunk["day_of_week"] = (
        chunk["timestamp"]
        .dt.dayofweek
        .astype("int8")
    )

    chunk["day_of_month"] = (
        chunk["timestamp"]
        .dt.day
        .astype("int8")
    )

    # --------------------------------------------------------
    # Log amount representations
    # --------------------------------------------------------

    chunk["amount_paid_log"] = np.log1p(
        chunk["amount_paid"]
    )

    chunk["amount_received_log"] = np.log1p(
        chunk["amount_received"]
    )

    # --------------------------------------------------------
    # Same-currency amount delta
    # --------------------------------------------------------

    same_currency_mask = (
        chunk["payment_currency"]
        == chunk["receiving_currency"]
    )

    chunk["same_currency_amount_delta"] = np.nan

    chunk.loc[
        same_currency_mask,
        "same_currency_amount_delta"
    ] = (
        chunk.loc[
            same_currency_mask,
            "amount_paid"
        ]
        -
        chunk.loc[
            same_currency_mask,
            "amount_received"
        ]
    )

    # --------------------------------------------------------
    # Final column order
    # --------------------------------------------------------

    final_columns = [
        "transaction_id",
        "timestamp",

        "from_bank",
        "from_account",
        "sender_id",

        "to_bank",
        "to_account",
        "receiver_id",

        "amount_paid",
        "amount_paid_log",
        "payment_currency",

        "amount_received",
        "amount_received_log",
        "receiving_currency",

        "same_currency_amount_delta",

        "payment_format",

        "is_cross_bank",
        "is_currency_conversion",
        "is_self_transfer",

        "hour",
        "day_of_week",
        "day_of_month",

        "is_laundering",
    ]

    return chunk[final_columns]


# ============================================================
# Validation
# ============================================================

def validate_processed_chunk(
    chunk: pd.DataFrame
) -> None:

    if chunk["transaction_id"].isna().any():
        raise ValueError(
            "Null transaction_id detected."
        )

    if chunk["timestamp"].isna().any():
        raise ValueError(
            "Invalid timestamp detected."
        )

    if chunk["sender_id"].isna().any():
        raise ValueError(
            "Null sender_id detected."
        )

    if chunk["receiver_id"].isna().any():
        raise ValueError(
            "Null receiver_id detected."
        )

    if (chunk["amount_paid"] < 0).any():
        raise ValueError(
            "Negative amount_paid detected."
        )

    if (chunk["amount_received"] < 0).any():
        raise ValueError(
            "Negative amount_received detected."
        )

    invalid_labels = (
        ~chunk["is_laundering"]
        .isin([0, 1])
    )

    if invalid_labels.any():
        raise ValueError(
            "Unexpected laundering label detected."
        )


# ============================================================
# Main pipeline
# ============================================================

def preprocess_transactions() -> dict:

    total_rows = 0
    laundering_rows = 0
    parts_written = 0

    minimum_timestamp = None
    maximum_timestamp = None

    reader = pd.read_csv(
        SOURCE_FILE,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    )

    for part_number, raw_chunk in enumerate(
        tqdm(
            reader,
            desc="Preprocessing transactions",
        )
    ):

        processed = normalize_chunk(
            raw_chunk,
            row_offset=total_rows,
        )

        validate_processed_chunk(
            processed
        )

        part_path = (
            TRANSACTION_OUTPUT_DIR
            / f"part-{part_number:05d}.parquet"
        )

        table = pa.Table.from_pandas(
            processed,
            preserve_index=False,
        )

        pq.write_table(
            table,
            part_path,
            compression="zstd",
            use_dictionary=True,
        )

        total_rows += len(processed)

        laundering_rows += int(
            processed["is_laundering"].sum()
        )

        chunk_min = (
            processed["timestamp"].min()
        )

        chunk_max = (
            processed["timestamp"].max()
        )

        if (
            minimum_timestamp is None
            or chunk_min < minimum_timestamp
        ):
            minimum_timestamp = chunk_min

        if (
            maximum_timestamp is None
            or chunk_max > maximum_timestamp
        ):
            maximum_timestamp = chunk_max

        parts_written += 1

    return {
        "dataset": "IBM AML HI-Small",
        "source_file": SOURCE_FILE.name,

        "rows": total_rows,

        "laundering_rows": laundering_rows,

        "laundering_rate": (
            laundering_rows / total_rows
            if total_rows
            else 0
        ),

        "parquet_parts": parts_written,

        "timestamp_start": (
            str(minimum_timestamp)
        ),

        "timestamp_end": (
            str(maximum_timestamp)
        ),
    }


# ============================================================
# Metadata
# ============================================================

def write_metadata(
    metadata: dict
) -> None:

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


def write_schema() -> None:

    first_file = next(
        TRANSACTION_OUTPUT_DIR.glob(
            "*.parquet"
        )
    )

    schema = pq.read_schema(
        first_file
    )

    schema_dict = {
        field.name: str(field.type)
        for field in schema
    }

    with open(
        SCHEMA_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            schema_dict,
            file,
            indent=2,
        )


# ============================================================
# Entry point
# ============================================================

def main() -> None:

    print(
        "\nIBM AML HI-Small "
        "Preprocessing Pipeline\n"
    )

    validate_source()

    prepare_output_directory()

    metadata = preprocess_transactions()

    write_metadata(metadata)

    write_schema()

    print("\nPreprocessing complete.")

    print(
        f"Rows processed: "
        f"{metadata['rows']:,}"
    )

    print(
        f"Laundering rows: "
        f"{metadata['laundering_rows']:,}"
    )

    print(
        f"Parquet parts: "
        f"{metadata['parquet_parts']}"
    )

    print(
        f"\nOutput:\n"
        f"{TRANSACTION_OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()