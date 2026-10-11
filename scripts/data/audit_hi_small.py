from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "ibm_aml" / "hi_small"
RESULTS_DIR = PROJECT_ROOT / "results"

REPORT_DIR = RESULTS_DIR / "reports"
PLOT_DIR = RESULTS_DIR / "plots"

TRANSACTIONS_FILE = RAW_DIR / "HI-Small_Trans.csv"
ACCOUNTS_FILE = RAW_DIR / "HI-Small_accounts.csv"
PATTERNS_FILE = RAW_DIR / "HI-Small_Patterns.txt"

REPORT_FILE = REPORT_DIR / "hi_small_dataset_profile.json"

CHUNK_SIZE = 500_000


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def ensure_directories() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)


def validate_source_files() -> None:
    if not TRANSACTIONS_FILE.exists():
        raise FileNotFoundError(
            f"Missing transaction file:\n{TRANSACTIONS_FILE}"
        )

    print(f"Transaction file found: {TRANSACTIONS_FILE}")

    if ACCOUNTS_FILE.exists():
        print(f"Accounts file found: {ACCOUNTS_FILE}")
    else:
        print("Accounts file not found — continuing without it.")

    if PATTERNS_FILE.exists():
        print(f"Patterns file found: {PATTERNS_FILE}")
    else:
        print("Patterns file not found — continuing without it.")


def safe_number(value):
    if pd.isna(value):
        return None

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        return float(value)

    return value


# ---------------------------------------------------------------------
# Transaction audit
# ---------------------------------------------------------------------

def audit_transactions() -> dict:
    print("\nAuditing HI-Small transaction dataset...")

    header = pd.read_csv(TRANSACTIONS_FILE, nrows=0)

    columns = list(header.columns)

    print("\nColumns:")
    for col in columns:
        print(f"  - {col}")

    total_rows = 0

    laundering_count = 0
    legitimate_count = 0

    missing_counts = Counter()

    payment_formats = Counter()
    receiving_currencies = Counter()
    payment_currencies = Counter()

    from_banks = set()
    to_banks = set()

    sender_accounts = set()
    receiver_accounts = set()

    min_timestamp = None
    max_timestamp = None

    min_amount_paid = None
    max_amount_paid = None
    sum_amount_paid = 0.0
    amount_paid_count = 0

    min_amount_received = None
    max_amount_received = None
    sum_amount_received = 0.0
    amount_received_count = 0

    self_transfer_count = 0
    cross_bank_count = 0
    currency_conversion_count = 0

    duplicate_count = 0

    # Used only for plotting
    sampled_amounts = []
    transactions_by_date = Counter()

    reader = pd.read_csv(
        TRANSACTIONS_FILE,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    )

    for chunk_number, chunk in enumerate(tqdm(reader, desc="Processing chunks")):

        total_rows += len(chunk)

        # -------------------------------------------------------------
        # Missing values
        # -------------------------------------------------------------

        for column, count in chunk.isna().sum().items():
            missing_counts[column] += int(count)

        # -------------------------------------------------------------
        # Label distribution
        # -------------------------------------------------------------

        if "Is Laundering" in chunk.columns:
            label_counts = chunk["Is Laundering"].value_counts(dropna=False)

            laundering_count += int(label_counts.get(1, 0))
            legitimate_count += int(label_counts.get(0, 0))

        # -------------------------------------------------------------
        # Categories
        # -------------------------------------------------------------

        if "Payment Format" in chunk.columns:
            payment_formats.update(
                chunk["Payment Format"]
                .dropna()
                .astype(str)
                .value_counts()
                .to_dict()
            )

        if "Receiving Currency" in chunk.columns:
            receiving_currencies.update(
                chunk["Receiving Currency"]
                .dropna()
                .astype(str)
                .value_counts()
                .to_dict()
            )

        if "Payment Currency" in chunk.columns:
            payment_currencies.update(
                chunk["Payment Currency"]
                .dropna()
                .astype(str)
                .value_counts()
                .to_dict()
            )

        # -------------------------------------------------------------
        # Bank / account uniqueness
        # -------------------------------------------------------------

        if "From Bank" in chunk.columns:
            from_banks.update(chunk["From Bank"].dropna().astype(str))

        if "To Bank" in chunk.columns:
            to_banks.update(chunk["To Bank"].dropna().astype(str))

        if {"From Bank", "Account"}.issubset(chunk.columns):
            sender_accounts.update(
                zip(
                    chunk["From Bank"].astype(str),
                    chunk["Account"].astype(str),
                )
            )

        if {"To Bank", "Account.1"}.issubset(chunk.columns):
            receiver_accounts.update(
                zip(
                    chunk["To Bank"].astype(str),
                    chunk["Account.1"].astype(str),
                )
            )

        # -------------------------------------------------------------
        # Timestamp analysis
        # -------------------------------------------------------------

        if "Timestamp" in chunk.columns:
            timestamps = pd.to_datetime(
                chunk["Timestamp"],
                errors="coerce",
            )

            chunk_min = timestamps.min()
            chunk_max = timestamps.max()

            if pd.notna(chunk_min):
                if min_timestamp is None or chunk_min < min_timestamp:
                    min_timestamp = chunk_min

            if pd.notna(chunk_max):
                if max_timestamp is None or chunk_max > max_timestamp:
                    max_timestamp = chunk_max

            date_counts = timestamps.dt.date.value_counts()

            for date, count in date_counts.items():
                transactions_by_date[str(date)] += int(count)

        # -------------------------------------------------------------
        # Amount statistics
        # -------------------------------------------------------------

        if "Amount Paid" in chunk.columns:
            amounts = pd.to_numeric(
                chunk["Amount Paid"],
                errors="coerce",
            ).dropna()

            if len(amounts):
                local_min = amounts.min()
                local_max = amounts.max()

                min_amount_paid = (
                    local_min
                    if min_amount_paid is None
                    else min(min_amount_paid, local_min)
                )

                max_amount_paid = (
                    local_max
                    if max_amount_paid is None
                    else max(max_amount_paid, local_max)
                )

                sum_amount_paid += amounts.sum()
                amount_paid_count += len(amounts)

                # small random sample for plotting
                sampled_amounts.extend(
                    amounts.sample(
                        min(1000, len(amounts)),
                        random_state=42,
                    ).tolist()
                )

        if "Amount Received" in chunk.columns:
            amounts = pd.to_numeric(
                chunk["Amount Received"],
                errors="coerce",
            ).dropna()

            if len(amounts):
                local_min = amounts.min()
                local_max = amounts.max()

                min_amount_received = (
                    local_min
                    if min_amount_received is None
                    else min(min_amount_received, local_min)
                )

                max_amount_received = (
                    local_max
                    if max_amount_received is None
                    else max(max_amount_received, local_max)
                )

                sum_amount_received += amounts.sum()
                amount_received_count += len(amounts)

        # -------------------------------------------------------------
        # Transaction behaviour sanity checks
        # -------------------------------------------------------------

        if {"From Bank", "Account", "To Bank", "Account.1"}.issubset(chunk.columns):

            sender = (
                chunk["From Bank"].astype(str)
                + "_"
                + chunk["Account"].astype(str)
            )

            receiver = (
                chunk["To Bank"].astype(str)
                + "_"
                + chunk["Account.1"].astype(str)
            )

            self_transfer_count += int((sender == receiver).sum())

        if {"From Bank", "To Bank"}.issubset(chunk.columns):
            cross_bank_count += int(
                (chunk["From Bank"] != chunk["To Bank"]).sum()
            )

        if {"Payment Currency", "Receiving Currency"}.issubset(chunk.columns):
            currency_conversion_count += int(
                (
                    chunk["Payment Currency"]
                    != chunk["Receiving Currency"]
                ).sum()
            )

        # Exact duplicates within each chunk.
        # This does not detect duplicates spanning different chunks.
        duplicate_count += int(chunk.duplicated().sum())

    # -----------------------------------------------------------------
    # Derived metrics
    # -----------------------------------------------------------------

    laundering_rate = (
        laundering_count / total_rows
        if total_rows
        else 0
    )

    avg_amount_paid = (
        sum_amount_paid / amount_paid_count
        if amount_paid_count
        else None
    )

    avg_amount_received = (
        sum_amount_received / amount_received_count
        if amount_received_count
        else None
    )

    all_accounts = sender_accounts | receiver_accounts

    return {
        "dataset": "IBM AML HI-Small",
        "columns": columns,
        "row_count": total_rows,

        "class_distribution": {
            "legitimate": legitimate_count,
            "laundering": laundering_count,
            "laundering_rate": laundering_rate,
        },

        "time_range": {
            "start": str(min_timestamp) if min_timestamp is not None else None,
            "end": str(max_timestamp) if max_timestamp is not None else None,
        },

        "amount_paid": {
            "min": safe_number(min_amount_paid),
            "max": safe_number(max_amount_paid),
            "mean": safe_number(avg_amount_paid),
        },

        "amount_received": {
            "min": safe_number(min_amount_received),
            "max": safe_number(max_amount_received),
            "mean": safe_number(avg_amount_received),
        },

        "entities": {
            "unique_sender_accounts": len(sender_accounts),
            "unique_receiver_accounts": len(receiver_accounts),
            "unique_accounts_total": len(all_accounts),
            "unique_from_banks": len(from_banks),
            "unique_to_banks": len(to_banks),
        },

        "behaviour": {
            "self_transfer_count": self_transfer_count,
            "cross_bank_count": cross_bank_count,
            "currency_conversion_count": currency_conversion_count,
        },

        "categorical_distributions": {
            "payment_formats": dict(payment_formats),
            "payment_currencies": dict(payment_currencies),
            "receiving_currencies": dict(receiving_currencies),
        },

        "missing_values": dict(missing_counts),

        "duplicate_rows_within_chunks": duplicate_count,

        "_plot_data": {
            "sampled_amounts": sampled_amounts,
            "transactions_by_date": dict(transactions_by_date),
        },
    }


# ---------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------

def create_plots(profile: dict) -> None:

    class_info = profile["class_distribution"]

    # -------------------------------------------------------------
    # Class distribution
    # -------------------------------------------------------------

    labels = ["Legitimate", "Laundering"]

    counts = [
        class_info["legitimate"],
        class_info["laundering"],
    ]

    plt.figure(figsize=(8, 5))
    plt.bar(labels, counts)
    plt.title("HI-Small Class Distribution")
    plt.ylabel("Transactions")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "class_distribution.png", dpi=150)
    plt.close()

    # -------------------------------------------------------------
    # Amount distribution
    # -------------------------------------------------------------

    amounts = profile["_plot_data"]["sampled_amounts"]

    if amounts:
        amounts = np.array(amounts)
        amounts = amounts[amounts > 0]

        plt.figure(figsize=(9, 5))
        plt.hist(np.log10(amounts), bins=80)
        plt.xlabel("log10(Amount Paid)")
        plt.ylabel("Frequency")
        plt.title("HI-Small Transaction Amount Distribution")
        plt.tight_layout()
        plt.savefig(
            PLOT_DIR / "transaction_amount_distribution.png",
            dpi=150,
        )
        plt.close()

    # -------------------------------------------------------------
    # Transactions over time
    # -------------------------------------------------------------

    date_counts = profile["_plot_data"]["transactions_by_date"]

    if date_counts:
        series = pd.Series(date_counts)
        series.index = pd.to_datetime(series.index)
        series = series.sort_index()

        plt.figure(figsize=(11, 5))
        plt.plot(series.index.to_numpy(), series.to_numpy())
        plt.xlabel("Date")
        plt.ylabel("Transactions")
        plt.title("HI-Small Transactions Over Time")
        plt.tight_layout()
        plt.savefig(
            PLOT_DIR / "transactions_over_time.png",
            dpi=150,
        )
        plt.close()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:

    ensure_directories()
    validate_source_files()

    profile = audit_transactions()

    create_plots(profile)

    # Remove temporary plot material before JSON export
    profile.pop("_plot_data", None)

    with open(REPORT_FILE, "w", encoding="utf-8") as file:
        json.dump(profile, file, indent=2)

    print("\nAudit complete.")
    print(f"Report: {REPORT_FILE}")
    print(f"Plots:  {PLOT_DIR}")

    print("\nKey statistics:")
    print(f"Transactions: {profile['row_count']:,}")
    print(
        f"Laundering: "
        f"{profile['class_distribution']['laundering']:,}"
    )
    print(
        f"Laundering rate: "
        f"{profile['class_distribution']['laundering_rate']:.6%}"
    )
    print(
        f"Unique accounts: "
        f"{profile['entities']['unique_accounts_total']:,}"
    )


if __name__ == "__main__":
    main()