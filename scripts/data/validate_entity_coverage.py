from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRANSACTION_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "hi_small"
    / "transactions"
)

ACCOUNTS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "hi_small"
    / "accounts.parquet"
)

REPORT_FILE = (
    PROJECT_ROOT
    / "results"
    / "reports"
    / "entity_coverage.json"
)


def main() -> None:

    print("\nValidating entity coverage...\n")

    accounts = pd.read_parquet(
        ACCOUNTS_FILE,
        columns=["account_id"],
    )

    known_accounts = set(
        accounts["account_id"]
    )

    transaction_dataset = ds.dataset(
        TRANSACTION_DIR,
        format="parquet",
    )

    table = transaction_dataset.to_table(
        columns=[
            "sender_id",
            "receiver_id",
        ]
    )

    sender_ids = set(
        table["sender_id"]
        .to_pylist()
    )

    receiver_ids = set(
        table["receiver_id"]
        .to_pylist()
    )

    transaction_accounts = (
        sender_ids | receiver_ids
    )

    covered = (
        transaction_accounts
        & known_accounts
    )

    missing = (
        transaction_accounts
        - known_accounts
    )

    coverage_rate = (
        len(covered)
        / len(transaction_accounts)
        if transaction_accounts
        else 0
    )

    report = {
        "transaction_accounts": len(
            transaction_accounts
        ),

        "accounts_metadata_rows": len(
            known_accounts
        ),

        "covered_accounts": len(
            covered
        ),

        "missing_accounts": len(
            missing
        ),

        "coverage_rate": coverage_rate,

        "missing_account_examples": list(
            sorted(missing)
        )[:50],
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
        f"Transaction accounts: "
        f"{len(transaction_accounts):,}"
    )

    print(
        f"Accounts in metadata: "
        f"{len(known_accounts):,}"
    )

    print(
        f"Covered: "
        f"{len(covered):,}"
    )

    print(
        f"Missing: "
        f"{len(missing):,}"
    )

    print(
        f"Coverage: "
        f"{coverage_rate:.4%}"
    )

    print(
        f"\nReport:\n"
        f"{REPORT_FILE}"
    )


if __name__ == "__main__":
    main()