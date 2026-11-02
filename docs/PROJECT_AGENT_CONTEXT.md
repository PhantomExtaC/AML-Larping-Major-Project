# FinCrime Intelligence — engineering context

## Purpose and current reality

This repository is a financial-crime detection platform whose implemented core is an IBM AML **HI-Small** research pipeline. It produces leakage-aware transaction features, chronological model splits, two gradient-boosting baselines, operational threshold analysis, and GraphSAGE-ready graph files. The README describes a larger intended product (FastAPI, PostgreSQL, Neo4j, frontend, investigations, sanctions, etc.); those application folders currently contain only `.gitkeep` placeholders. Do not describe those planned services as implemented.

The target is binary transaction/edge classification:

`is_laundering = 1` means a suspicious/laundering transaction.

The raw IBM AML data is deliberately ignored by Git and must be treated as source data, not edited or exhaustively inspected. All derived data uses Parquet with Zstandard compression where written by Polars.

## Repository map

```text
Majorproject/
├── README.md                         # broad platform vision and intended architecture
├── pyproject.toml                    # package metadata; Python >=3.13
├── .env.example                      # future API/database/Neo4j settings
├── requirements/
│   ├── base.txt                      # FastAPI, Pydantic, SQLAlchemy, PostgreSQL driver
│   ├── research.txt                  # pandas, numpy, pyarrow, polars, sklearn, XGBoost, LightGBM, MLflow, Optuna
│   ├── graph.txt                     # networkx, torch, torch-geometric
│   └── dev.txt                       # pytest, ruff, black, mypy, notebook tooling
├── data/                             # ignored local datasets and derived artifacts
│   ├── raw/ibm_aml/hi_small/         # CSV source: transactions, accounts, patterns
│   └── processed/hi_small/           # canonical Parquet, feature sets, splits, graph exports
├── scripts/
│   ├── data/                         # preprocessing, feature engineering, validation, inspection
│   ├── train/                        # shared tabular contract + LightGBM/XGBoost training
│   ├── evaluate/                     # threshold and top-K analysis
│   └── graph/                        # GraphSAGE v1 export
├── research/
│   ├── configs/, experiments/, notebooks/, outputs/  # reserved, currently empty placeholders
│   └── models/baseline/              # persisted model binaries and feature contract
├── results/
│   ├── reports/                      # data/split/validation reports
│   ├── plots/                        # initial data-audit charts
│   ├── models/                       # validation predictions, metrics, feature importances
│   └── comparisons/threshold_analysis/ # operating-point comparisons
├── backend/, frontend/, infrastructure/, shared/     # planned only; placeholders
└── amlPlayground/                    # local Python virtual environment; not application source
```

## Dataset and canonical data contract

### Raw inputs

`data/raw/ibm_aml/hi_small/` contains:

- `HI-Small_Trans.csv`: 5,078,345 transaction rows, 5,177 positives (0.10194%). Time range: 2022-09-01 00:00 through 2022-09-18 16:18.
- `HI-Small_accounts.csv`: account metadata.
- `HI-Small_Patterns.txt`: supplied pattern reference; it is present as raw reference material but not consumed by the implemented training pipeline.

Raw transaction schema, in exact source order:

`Timestamp, From Bank, Account, To Bank, Account.1, Amount Received, Receiving Currency, Amount Paid, Payment Currency, Payment Format, Is Laundering`.

### Phase 1 normalization

`scripts/data/preprocess_hi_small.py` reads the transaction CSV in 500,000-row pandas chunks and writes 11 Parquet partitions under `data/processed/hi_small/transactions/`. It validates the source header, assigns a stable sequential `transaction_id`, normalizes IDs, parses timestamps, adds static base fields, validates every processed chunk, and writes `metadata.json` and `schema.json`.

Canonical transaction fields are:

- identifiers: `transaction_id`, `timestamp`, `from_bank`, `from_account`, `sender_id`, `to_bank`, `to_account`, `receiver_id`
- amounts/currency: `amount_paid`, `amount_paid_log`, `payment_currency`, `amount_received`, `amount_received_log`, `receiving_currency`, `same_currency_amount_delta`
- transaction context: `payment_format`, `is_cross_bank`, `is_currency_conversion`, `is_self_transfer`, `hour`, `day_of_week`, `day_of_month`
- label: `is_laundering`

`sender_id` and `receiver_id` are the canonical account keys, based on bank/account identity. The account preprocessor (`scripts/data/preprocess_hi_small_accounts.py`) creates `data/processed/hi_small/accounts.parquet`, derives `entity_type` from synthetic entity names, and reports 518,581 unique accounts across 30,470 banks. Coverage validation proves every one of the 515,088 transaction-seen accounts exists in metadata.

## Feature pipeline

### Non-negotiable temporal rule

All historical features are **strictly prior-time**. Same-timestamp events are simultaneous and cannot see each other. Phase 2B rolling operations use `closed='left'`. This is the central leakage guardrail: preserve it in every future transformation, model experiment, and online implementation.

### Phase 2A: static and account-history features

Implemented in `scripts/data/build_features_phase2a.py`; produces `data/processed/hi_small/modeling/phase2a/features.parquet` with 49 columns.

- Static: log amounts, same-currency amount delta, cross-bank/conversion/self-transfer flags, calendar fields, and cyclic hour/day-of-week encodings.
- Unified account history (sender and receiver independently): prior activity count, observed tenure in hours/days, seconds since prior activity, and new-account flags at 1h/24h/3d.
- Directional history: sender prior outgoing count/gap and receiver prior incoming count/gap.
- Metadata: sender/receiver entity type.

Observed tenure is only time since an account's first appearance in this dataset—not real account age. Missing time-since-previous means no prior event; it must not be silently replaced with zero.

### Phase 2B: behavioural features

Implemented in `scripts/data/build_features_phase2b.py`; produces `data/processed/hi_small/modeling/phase2b/features.parquet` with 125 columns. It supports `--stage {all,1,2,3}` and persists intermediate stage Parquets under `phase2b/stages/`.

1. **Stage 1: velocity and volume.** Sender and receiver trailing transaction counts for 10m, 1h, 6h, 24h, 3d; trailing amount sums for 1h, 6h, 24h, 3d.
2. **Stage 2: counterparties.** Unique receivers/senders over 1h and 24h; sender→receiver prior transaction count; `is_new_counterparty`.
3. **Stage 3: flow and amount deviation.** Trailing inbound/outbound amounts, 24h rolling means and standard deviations, amount-to-mean ratio, z-score, inflow/outflow ratio, normalized flow balance, pass-through ratio, and fund retention.

The Phase 2B validator confirms all 5,078,345 rows and 5,177 labels survive; window nesting/counterparty bounds/lifetime checks have zero violations. Nullable deviation fields are expected when no usable historical denominator exists. The label is carried through but never participates in feature construction.

### Current modelling feature contract

`scripts/train/tabular_common.py` discovers numeric columns from the **training split only**, excludes target/IDs/raw account columns/entity type and helper statistics (`*_amount_sumsq_*`, `*_amount_count_*`), then one-hot encodes three categoricals from train-only categories. Unknown future categories become all-zero.

- 95 numeric features + 37 one-hot fields = **132 input features**.
- Categoricals: `payment_format` (7), `payment_currency` (15), `receiving_currency` (15).
- Matrices are `float32`; `±inf` is changed to `NaN` because the tree libraries handle missing values natively.
- `feature_definition.json` is the persisted scoring contract. Do not reorder, casually rename, or regenerate it for an existing model.

Feature families in the 132-field model: transaction amounts/context/time; observed tenure and activity gaps; sender/receiver velocity and volume; counterparty novelty; flow; amount deviation; and three categorical one-hot groups. Raw account/bank identifiers and synthetic entity type are deliberately excluded to limit memorization and simulator leakage.

## Split protocol

`scripts/data/create_model_splits.py` freezes chronological splits from Phase 2B features. It refuses to split identical timestamps and preserves the late period as a separate stress distribution.

| Split | Period | Rows | Positives | Positive rate | Role |
|---|---:|---:|---:|---:|---|
| train | Sep 1 00:00–Sep 7 14:51 | 3,553,739 | 2,856 | 0.0804% | fit models/features |
| validation | Sep 7 14:52–Sep 9 03:13 | 761,646 | 760 | 0.0998% | early stopping and operating-point selection |
| test | Sep 9 03:14–Sep 10 23:59 | 761,852 | 906 | 0.1189% | frozen ordinary holdout; not yet reported by stored baselines |
| tail_stress | Sep 11 00:05–Sep 18 16:18 | 1,108 | 655 | 59.12% | extreme late-period distribution-shift stress set |

The dense benchmark targets 70% train and 85% cumulative train+validation before timestamp adjustment. Tail stress is not a realistic prevalence sample, so operational performance there must be interpreted cautiously.

## Models and current results

Both current baselines calculate `scale_pos_weight = negatives / positives = 1243.3064` from the training data. They train with the 132-feature contract and evaluate on validation; prediction files retain `transaction_id`, `true_label`, and model probability.

### XGBoost (`xgboost_f4_v1`) — preferred baseline

Source: `scripts/train/train_xgboost.py`; model: `research/models/baseline/xgboost_f4_v1/model.json`.

Parameters: `objective=binary:logistic`, `eval_metric=aucpr`, `tree_method=hist`, learning rate 0.05, max depth 6, min child weight 5, subsample 0.80, column sample 0.80, L1 0.10, L2 1.0, max bin 256, seed 42, all available CPU threads, up to 3,000 rounds, early stopping after 100 rounds. Best iteration: **1052**.

Validation: PR-AUC **0.42039**, ROC-AUC **0.98047**, best F1 **0.46359** at threshold **0.95896** (precision 0.57253, recall 0.38947). Top-100 alerts find 97 true positives (97% precision; 12.76% recall); top-10,000 finds 569 (5.69% precision; 74.87% recall). This is the model to extend or benchmark against.

Leading gain features include `payment_format__ach`, `is_new_counterparty`, sender 6h/1h velocity, sender unique receivers, Saudi Riyal/Bitcoin indicators, self/cross-bank flags, and sender-receiver prior count. These importances are associations in a synthetic benchmark, not causal explanations.

### LightGBM (`lightgbm_f4_v1`) — retained as a weak comparison

Source: `scripts/train/train_lightgbm.py`; model: `research/models/baseline/lightgbm_f4_v1/model.txt`.

Parameters: binary objective, average-precision metric, learning rate 0.05, 63 leaves, unlimited depth, minimum leaf 100, feature/bagging fractions 0.8, bagging frequency 1, L1 0.1, L2 1.0, max bin 255, all random seeds 42, all threads. Training allows 3,000 rounds with 100-round early stopping. Best iteration: **1**.

Validation: PR-AUC **0.01795**, ROC-AUC **0.95208**, best F1 **0.03938** at a threshold displayed as 1.0. It is not production-worthy and should be treated as a failed/diagnostic baseline, not an ensemble partner.

### Threshold evaluation

`scripts/evaluate/threshold_analysis.py` produces complete threshold tables, recall-target selections (80/90/95/99%), top-K metrics (100/500/1,000/5,000/10,000), and comparison CSV/JSON files from stored validation predictions. Never promote a threshold based only on ROC-AUC; use precision/recall, analyst alert capacity, and false positives per true positive.

For XGBoost, 80% recall uses threshold 0.03219 and creates 19,507 alerts (3.12% precision); 95% recall uses 0.001405 and creates 76,034 alerts (0.95% precision). The high best-F1 threshold is a precision-oriented operating point, not a universal decision threshold.

## Graph systems

Two graph export generations exist in ignored processed data; **GraphSAGE v1 is the current code-backed export**.

`scripts/graph/export_gnn_dataset.py` creates a directed account-to-account transaction graph under `data/processed/hi_small/graph_v1/`:

- 518,581 nodes, indexed deterministically by sorted `account_id`.
- 12 node features are train-only aggregates: outgoing/incoming counts, amount sums/means, unique counterparties, total activity/counterparties, net flow, normalized flow balance.
- Edges are transactions with `src`, `dst`, timestamp, transaction ID, 35 leakage-safe Phase 2B edge features, and `is_laundering`.
- Separate train/validation/test/tail edge files preserve the tabular split boundaries; all edge endpoints map successfully.
- Training topology contains only training-period edges. Do not leak validation/test connectivity into a graph model's training message-passing graph unless explicitly designing a transductive experiment and documenting that choice.

`data/processed/hi_small/graph/graph_metadata.json` is an older, simpler graph export with raw/static edge features. Prefer `graph_v1` for new work.

No GNN is trained in this repository yet; `requirements/graph.txt` provides the intended libraries only.

## Script run order and safeguards

```text
audit_hi_small.py                         # profile raw source + plots
preprocess_hi_small.py                    # raw transactions -> canonical Parquet
preprocess_hi_small_accounts.py           # raw accounts -> accounts Parquet
validate_entity_coverage.py               # confirms transaction/account join coverage
build_features_phase2a.py                 # canonical -> history/static features
validate_features_phase2a.py
build_features_phase2b.py --stage all     # behavioural features
validate_features_phase2b.py
create_model_splits.py                    # immutable chronological Parquets
train_xgboost.py                          # recommended baseline rerun
train_lightgbm.py                         # comparison only
threshold_analysis.py                     # operational validation analysis
export_gnn_dataset.py                     # GraphSAGE v1 dataset
```

Inspection helpers: `inspect_features_phase2b.py <transaction_id>` and `inspect_account_history.py <account_id>`. Most write scripts delete/recreate their own exact output artifact before writing; do not run them blindly if preserving an alternative output is important.

## Architecture status, boundaries, and next steps

- Implemented: data engineering, feature engineering, validation, frozen time splits, boosted-tree training, offline operational evaluation, graph-data export.
- Planned only: FastAPI APIs, database schemas/migrations, Neo4j integration, frontend, case management, model service, SHAP explanation service, MLflow experiment tracking in code, classical graph analytics, and any deployed real-time feature store.
- The `.env.example` names a future PostgreSQL database, entity hash secret, and Neo4j connection; no code currently consumes these settings.
- Do not claim test or tail-stress metrics exist: the stored trained-model reports are validation-only.
- When adding experiments, preserve chronological evaluation; fit category vocabularies/scalers/imputers on training only; never use label-derived features; record model artifact, exact feature definition, seed, split, metrics, and Git revision.
- The primary engineering gap for real-time scoring is a stateful, strictly prior-time account/counterparty feature store that reproduces Phase 2B semantics at event time.

## Important artifacts

- `research/models/baseline/xgboost_f4_v1/feature_definition.json`: exact 132-feature scoring schema.
- `results/models/xgboost_f4_v1/validation_metrics.json`: authoritative stored baseline summary.
- `results/comparisons/threshold_analysis/operational_summary.json`: concise validation operating-point comparison.
- `results/reports/phase2a_validation.json`, `phase2b_validation.json`, `split_profile.json`: pipeline integrity and temporal-split evidence.
- `docs/dataset_manifest.md`: currently only a placeholder; this document is the maintained project-level agent context.
