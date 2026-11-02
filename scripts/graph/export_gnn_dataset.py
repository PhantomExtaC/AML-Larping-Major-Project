from __future__ import annotations

import json
from pathlib import Path

import polars as pl


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

ACCOUNTS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "hi_small"
    / "accounts.parquet"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "hi_small"
    / "graph_v1"
)

NODES_FILE = (
    OUTPUT_DIR
    / "nodes.parquet"
)

METADATA_FILE = (
    OUTPUT_DIR
    / "graph_metadata.json"
)


# ============================================================
# Graph feature contract
# ============================================================

EDGE_FEATURES_V1 = [
    # --------------------------------------------------------
    # Current transaction
    # --------------------------------------------------------

    "amount_paid_log",
    "amount_received_log",

    "is_cross_bank",
    "is_currency_conversion",
    "is_self_transfer",

    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",

    # --------------------------------------------------------
    # Velocity
    # --------------------------------------------------------

    "sender_tx_count_10m",
    "sender_tx_count_1h",
    "sender_tx_count_6h",
    "sender_tx_count_24h",

    "receiver_tx_count_10m",
    "receiver_tx_count_1h",
    "receiver_tx_count_6h",
    "receiver_tx_count_24h",

    # --------------------------------------------------------
    # Counterparty behaviour
    # --------------------------------------------------------

    "sender_unique_receivers_1h",
    "sender_unique_receivers_24h",

    "receiver_unique_senders_1h",
    "receiver_unique_senders_24h",

    "sender_receiver_prior_tx_count",
    "is_new_counterparty",

    # --------------------------------------------------------
    # Behaviour deviation
    # --------------------------------------------------------

    "sender_amount_ratio_to_mean_24h",
    "sender_amount_zscore_24h",

    "receiver_amount_ratio_to_mean_24h",
    "receiver_amount_zscore_24h",

    # --------------------------------------------------------
    # Flow
    # --------------------------------------------------------

    "sender_flow_balance_1h",
    "sender_flow_balance_24h",

    "sender_pass_through_ratio_1h",
    "sender_pass_through_ratio_24h",

    "receiver_flow_balance_1h",
    "receiver_flow_balance_24h",

    "receiver_pass_through_ratio_1h",
    "receiver_pass_through_ratio_24h",
]


EDGE_COLUMNS = [
    "transaction_id",
    "timestamp",

    "sender_id",
    "receiver_id",

    *EDGE_FEATURES_V1,

    "is_laundering",
]


NODE_FEATURES = [
    "train_out_tx_count",
    "train_out_amount_sum",
    "train_out_amount_mean",
    "train_unique_receivers",

    "train_in_tx_count",
    "train_in_amount_sum",
    "train_in_amount_mean",
    "train_unique_senders",

    "train_total_tx_count",
    "train_total_counterparties",

    "train_net_flow",
    "train_flow_balance",
]


# ============================================================
# Helpers
# ============================================================

def print_step(message: str) -> None:
    print("\n" + "=" * 76)
    print(message)
    print("=" * 76)


def load_split(
    name: str,
) -> pl.LazyFrame:

    path = (
        SPLIT_DIR
        / f"{name}.parquet"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing split:\n{path}"
        )

    return (
        pl.scan_parquet(
            str(path)
        )
    )


# ============================================================
# Node table
# ============================================================

def build_node_features(
    train: pl.LazyFrame,
) -> pl.LazyFrame:
    """
    Build account-level node features from TRAIN behaviour only.

    No validation/test activity is used here.
    """

    outgoing = (
        train
        .group_by(
            "sender_id"
        )
        .agg(
            [
                pl.len()
                .alias(
                    "train_out_tx_count"
                ),

                pl.col(
                    "amount_paid"
                )
                .sum()
                .alias(
                    "train_out_amount_sum"
                ),

                pl.col(
                    "amount_paid"
                )
                .mean()
                .alias(
                    "train_out_amount_mean"
                ),

                pl.col(
                    "receiver_id"
                )
                .n_unique()
                .alias(
                    "train_unique_receivers"
                ),
            ]
        )
        .rename(
            {
                "sender_id":
                    "account_id"
            }
        )
    )

    incoming = (
        train
        .group_by(
            "receiver_id"
        )
        .agg(
            [
                pl.len()
                .alias(
                    "train_in_tx_count"
                ),

                pl.col(
                    "amount_received"
                )
                .sum()
                .alias(
                    "train_in_amount_sum"
                ),

                pl.col(
                    "amount_received"
                )
                .mean()
                .alias(
                    "train_in_amount_mean"
                ),

                pl.col(
                    "sender_id"
                )
                .n_unique()
                .alias(
                    "train_unique_senders"
                ),
            ]
        )
        .rename(
            {
                "receiver_id":
                    "account_id"
            }
        )
    )

    accounts = (
        pl.scan_parquet(
            str(ACCOUNTS_FILE)
        )
        .select(
            [
                "account_id",
                "entity_type",
            ]
        )
    )

    nodes = (
        accounts
        .join(
            outgoing,
            on="account_id",
            how="left",
        )
        .join(
            incoming,
            on="account_id",
            how="left",
        )
    )

    base_numeric = [
        "train_out_tx_count",
        "train_out_amount_sum",
        "train_out_amount_mean",
        "train_unique_receivers",

        "train_in_tx_count",
        "train_in_amount_sum",
        "train_in_amount_mean",
        "train_unique_senders",
    ]

    nodes = (
        nodes
        .with_columns(
            [
                pl.col(column)
                .fill_null(0)
                for column
                in base_numeric
            ]
        )
    )

    nodes = (
        nodes
        .with_columns(
            [
                (
                    pl.col(
                        "train_out_tx_count"
                    )
                    +
                    pl.col(
                        "train_in_tx_count"
                    )
                )
                .alias(
                    "train_total_tx_count"
                ),

                (
                    pl.col(
                        "train_unique_receivers"
                    )
                    +
                    pl.col(
                        "train_unique_senders"
                    )
                )
                .alias(
                    "train_total_counterparties"
                ),

                (
                    pl.col(
                        "train_in_amount_sum"
                    )
                    -
                    pl.col(
                        "train_out_amount_sum"
                    )
                )
                .alias(
                    "train_net_flow"
                ),

                (
                    (
                        pl.col(
                            "train_in_amount_sum"
                        )
                        -
                        pl.col(
                            "train_out_amount_sum"
                        )
                    )
                    /
                    (
                        pl.col(
                            "train_in_amount_sum"
                        )
                        +
                        pl.col(
                            "train_out_amount_sum"
                        )
                        +
                        1e-9
                    )
                )
                .alias(
                    "train_flow_balance"
                ),
            ]
        )
    )

    return (
        nodes
        .sort(
            "account_id"
        )
        .with_row_index(
            name="node_index",
            offset=0,
        )
    )


# ============================================================
# Edge export
# ============================================================

def export_edges(
    split_name: str,
    output_name: str,
    node_map: pl.LazyFrame,
) -> dict:

    split = (
        load_split(
            split_name
        )
        .select(
            EDGE_COLUMNS
        )
    )

    sender_map = (
        node_map
        .select(
            [
                pl.col(
                    "account_id"
                )
                .alias(
                    "sender_id"
                ),

                pl.col(
                    "node_index"
                )
                .alias(
                    "src"
                ),
            ]
        )
    )

    receiver_map = (
        node_map
        .select(
            [
                pl.col(
                    "account_id"
                )
                .alias(
                    "receiver_id"
                ),

                pl.col(
                    "node_index"
                )
                .alias(
                    "dst"
                ),
            ]
        )
    )

    edges = (
        split
        .join(
            sender_map,
            on="sender_id",
            how="left",
        )
        .join(
            receiver_map,
            on="receiver_id",
            how="left",
        )
    )

    output_columns = [
        "transaction_id",
        "timestamp",

        "src",
        "dst",

        "sender_id",
        "receiver_id",

        *EDGE_FEATURES_V1,

        "is_laundering",
    ]

    edges = (
        edges
        .select(
            output_columns
        )
    )

    output_path = (
        OUTPUT_DIR
        / output_name
    )

    if output_path.exists():
        output_path.unlink()

    edges.sink_parquet(
        str(output_path),
        compression="zstd",
        statistics=True,
        maintain_order=True,
        engine="streaming",
    )

    summary = (
        pl.scan_parquet(
            str(output_path)
        )
        .select(
            [
                pl.len()
                .alias(
                    "rows"
                ),

                pl.col(
                    "is_laundering"
                )
                .sum()
                .alias(
                    "positives"
                ),

                pl.col(
                    "src"
                )
                .is_null()
                .sum()
                .alias(
                    "missing_src"
                ),

                pl.col(
                    "dst"
                )
                .is_null()
                .sum()
                .alias(
                    "missing_dst"
                ),
            ]
        )
        .collect()
        .to_dicts()[0]
    )

    if (
        summary["missing_src"] > 0
        or summary["missing_dst"] > 0
    ):
        raise ValueError(
            f"Node mapping failure: "
            f"{summary}"
        )

    return summary


# ============================================================
# Main
# ============================================================

def main() -> None:

    print(
        "\n"
        "IBM AML HI-Small\n"
        "Graph Dataset Export — GraphSAGE v1\n"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train = (
        load_split(
            "train"
        )
    )

    # --------------------------------------------------------
    # Nodes
    # --------------------------------------------------------

    print_step(
        "Building train-only node features"
    )

    nodes = (
        build_node_features(
            train
        )
    )

    if NODES_FILE.exists():
        NODES_FILE.unlink()

    nodes.sink_parquet(
        str(NODES_FILE),
        compression="zstd",
        statistics=True,
        maintain_order=True,
        engine="streaming",
    )

    node_count = (
        pl.scan_parquet(
            str(NODES_FILE)
        )
        .select(
            pl.len()
        )
        .collect()
        .item()
    )

    print(
        f"Nodes: {node_count:,}"
    )

    node_map = (
        pl.scan_parquet(
            str(NODES_FILE)
        )
    )

    # --------------------------------------------------------
    # Edges
    # --------------------------------------------------------

    print_step(
        "Exporting GraphSAGE v1 edges"
    )

    definitions = [
        (
            "train",
            "train_edges.parquet",
        ),

        (
            "validation",
            "validation_edges.parquet",
        ),

        (
            "test",
            "test_edges.parquet",
        ),

        (
            "tail_stress",
            "tail_edges.parquet",
        ),
    ]

    summaries = {}

    for split_name, output_name in definitions:

        print(
            f"\n{split_name}..."
        )

        summaries[
            split_name
        ] = export_edges(
            split_name,
            output_name,
            node_map,
        )

        print(
            summaries[
                split_name
            ]
        )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = {
        "version":
            "graphsage_v1",

        "task":
            "transaction_edge_binary_classification",

        "target":
            "is_laundering",

        "num_nodes":
            node_count,

        "node_features":
            NODE_FEATURES,

        "edge_features":
            EDGE_FEATURES_V1,

        "edge_feature_count":
            len(
                EDGE_FEATURES_V1
            ),

        "node_feature_count":
            len(
                NODE_FEATURES
            ),

        "topology":
            (
                "Directed account-to-account transaction "
                "graph. Training topology contains only "
                "training-period edges."
            ),

        "node_feature_semantics":
            (
                "Node aggregates are calculated using "
                "training-period transactions only."
            ),

        "edge_feature_semantics":
            (
                "Phase 2B behavioral features were created "
                "with strict prior-time semantics."
            ),

        "splits":
            summaries,
    }

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

    print_step(
        "GRAPH V1 EXPORT COMPLETE"
    )

    print(
        f"Upload folder:\n"
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()