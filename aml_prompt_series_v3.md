# AML Platform: Sequential Prompt Series (v3)

**How to use this document**

1. Open a fresh chat with your coding chatbot. Paste **PART 0 (Project Charter)** first, as its own message. Wait for it to acknowledge.
2. Then paste the prompts **P1, P2, ... in order**, one per message. After each, run the code, and paste back errors or outputs. The bot should reply with only the changed files, in full.
3. Each prompt has a **"YOU PASTE"** box listing material the bot cannot see (offline feature code, JSON schemas, outputs of earlier steps). Paste it together with the prompt. The bot has no access to your repo, so do not skip these.
4. The GraphSAGE *improvement/retraining* is a separate, later prompt (placeholder at the end). **P7 builds the slot-in mechanism now**, so a newly trained model drops in without code changes.
5. If a prompt is too big for one reply, tell the bot "continue with the remaining files" and it should resume from the next file.

Suggested order and parallelism: P1 → P2 → P3 → P4 (needs P3) → P5 (independent of P2-P4; can be done any time after P1) → P6 → P7 → P8 → P9a → P9b → P10 → P11. The frontend (P9a/P9b) can start as soon as P8's API contract exists.

---

# PART 0: PROJECT CHARTER (paste once, first)

You are my implementation partner for a final-year Computer Engineering project. I will send you a series of build prompts, one at a time. For each one, you will write **complete, runnable code files**. I do the auditing and testing and will tell you when something breaks, so do **not** ask for repo audits or propose plans before coding. State at most 5 short assumptions, then write the code.

## 0.1 Project

**An Explainable, Multi-Agent AML Simulation and Risk Investigation Platform Using Temporal Transaction Intelligence and Graph-Based Escalation.**

A local sandbox that: (1) generates synthetic transactions from legitimate and suspicious behavioural agents, or replays the IBM AML HI-Small dataset; (2) computes strictly prior-time features online; (3) scores every transaction with a pretrained XGBoost model; (4) selectively escalates transactions to an asynchronous, bounded temporal graph investigation (graph analytics plus a pluggable GraphSAGE model); (5) explains the tabular score with SHAP and the graph result with evidence; (6) exposes everything via FastAPI and a React analyst dashboard. It is a simulation and validation sandbox. It never connects to real banks or moves real money. All data is synthetic.

**Research question:** does a selective, risk-triggered graph investigation layer add detection or analyst-context value over a strong tabular classifier, at acceptable cost? Do not assume graph models are better.

## 0.2 Environment

Windows, PowerShell, Python virtual environment `amlPlayground`, project root `C:\Projects\Majorproject`, Python >=3.13. Laptop: i5-13500H, 16 GB RAM, RTX 2050 4 GB. Inference must run on CPU. Heavy model training happens on Google Colab only. Give **PowerShell commands** for everything (install, run, test). Pin package versions you rely on in a `requirements/*.txt` snippet and keep dependencies minimal. No Docker, Kafka, Redis, Neo4j or Kubernetes.

## 0.3 What already exists (do not recreate)

- Data (git-ignored): `data/raw/ibm_aml/hi_small/`, `data/processed/hi_small/{transactions/, accounts.parquet, modeling/{phase2a,phase2b,splits}/, graph_v1/}`. Parquet with Zstandard.
- Splits (frozen, chronological, never reshuffle): train 3,553,739 rows (2,856 positives), validation 761,646 (760), test 761,852 (906, untouched until final evaluation), tail_stress 1,108 (655; a separate distribution-shift set, never used for tuning).
- Features: Phase 2A (49 cols) and Phase 2B (125 cols) are strictly prior-time (rolling windows closed on the left). The model contract is **132 inputs = 95 numeric + 37 one-hot** (`payment_format` 7, `payment_currency` 15, `receiving_currency` 15). Matrices are float32. `±inf` becomes `NaN`. Unknown categories become all-zero. Missing "time since previous" means no prior event and stays `NaN` (never 0).
- Authoritative scoring schema: `research/models/baseline/xgboost_f4_v1/feature_definition.json` (exact names and order). Never regenerate or reorder it.
- XGBoost model `xgboost_f4_v1`: `research/models/baseline/xgboost_f4_v1/model.json`. It was trained with `scale_pos_weight` ≈ 1243.3, so **scores are not calibrated probabilities**. Validation PR-AUC 0.4204, ROC-AUC 0.9805, best F1 0.4636 at threshold 0.95896. Stored validation predictions: `results/models/xgboost_f4_v1/validation_predictions.parquet` (columns `transaction_id`, `true_label`, probability).
- Threshold analysis already exists (`results/comparisons/threshold_analysis/`). At 80% recall the threshold is 0.03219 (19,507 alerts, precision 3.12%). At 95% recall it is 0.001405 (76,034 alerts, precision 0.95%). Top-100 alerts: 97 true positives.
- GraphSAGE exists **only on Google Colab** (current validation PR-AUC ≈ 0.092, much weaker than XGBoost). It will be retrained and improved later. Graph export: `data/processed/hi_small/graph_v1/` (518,581 nodes indexed by sorted `account_id`; 12 train-only node features; 35 edge features; train/validation/test/tail edge files). The current classifier input is 291 dimensions (probably 2×128 node embeddings + 35 edge features; I will confirm).
- Canonical transaction columns: `transaction_id, timestamp, from_bank, from_account, sender_id, to_bank, to_account, receiver_id, amount_paid, amount_paid_log, payment_currency, amount_received, amount_received_log, receiving_currency, same_currency_amount_delta, payment_format, is_cross_bank, is_currency_conversion, is_self_transfer, hour, day_of_week, day_of_month, is_laundering`.
- `backend/`, `frontend/` are empty placeholders. Everything in them is built by this series.

## 0.4 Non-negotiable rules

**Temporal correctness:** features use only events with `timestamp < current timestamp`. Events sharing a timestamp are processed as a batch: compute all their features from the pre-batch state, score them, then update state. The candidate transaction is never in its own history or graph context. Graph context never contains future edges.
**Leakage:** labels are never model inputs. Account IDs, bank names, entity names and entity type are not model features. Simulation ground truth lives in a separate stream and never enters any feature vector. The test split never influences thresholds or tuning.
**Honest claims:** the data is synthetic. Never claim GNN > XGBoost without results. Graph structure does not prove laundering. SHAP values are in log-odds space and are associations, not causes, and SHAP does not explain the GNN. Never fabricate metrics or test results.
**Reliability:** the first-stage decision must survive investigation failures. Every transaction gets a traceable processing record. Nothing is silently dropped. Cold-start status is explicit: `graph_available | graph_partially_available | graph_unavailable_cold_start`.
**Reproducibility:** seeded RNGs, versioned configs, model and feature versions in every output.

## 0.5 Repository layout and shared contracts (all prompts conform to this)

```text
backend/
  api/{main.py, routes/{simulation.py,transactions.py,alerts.py,investigations.py,models.py,metrics.py,stream.py}}
  core/{config.py, schemas.py, logging.py, ids.py}
  configs/{app.yaml, risk_policy.yaml, investigation.yaml, graph_model.yaml, scenarios/*.yaml}
  services/
    simulation/{account_registry.py, agents.py, scenarios.py, event_scheduler.py, transaction_generator.py, controller.py}
    features/{online_feature_engine.py, account_state.py, rolling_windows.py, feature_contract.py, snapshot.py}
    inference/{model_loader.py, xgboost_scorer.py, risk_policy.py}
    investigation/{temporal_graph_store.py, graph_builder.py, graph_analytics.py, graph_rules.py, escalation.py, investigation_service.py, job_queue.py}
    graph_models/{base.py, registry.py, manifest.py, node_features.py, architectures/{__init__.py, graphsage_edge_v1.py, sage_layers.py}, null_scorer.py, loader.py}
    explainability/{shap_service.py, global_importance.py, graph_evidence.py, report_builder.py}
    pipeline/{orchestrator.py, event_bus.py}
    storage/{db.py, alert_store.py, investigation_store.py, transaction_store.py}
  tests/
frontend/            # React + Vite + TypeScript
scripts/{data,train,evaluate,graph,utilities}/
research/models/{baseline,graph}/          # graph/<model_name>/ holds slot-in bundles (see P7)
results/{reports,plots,models,comparisons,metrics}/
docs/
```

**Core schema names (Pydantic v2, in `backend/core/schemas.py`):** `TransactionEvent`, `GroundTruthRecord`, `FeatureRow`, `DecisionRecord`, `GraphEdge`, `GraphNode`, `Subgraph`, `GraphScore`, `InvestigationResult`, `ShapExplanation`, `InvestigationReport`, `SimulationStatus`, `StreamMessage`. Whenever a later prompt needs to change a schema, **say so explicitly and output the full updated `schemas.py`**.

**Plug-in contract for any graph model:** `GraphScorer.score(subgraph, target_event, context) -> GraphScore`, described in P7. The orchestrator only knows this interface.

## 0.6 Response format (every prompt)

1. "Assumptions" (≤5 bullets).
2. **Every file complete**, never partial, with the path as a heading above each code block. No `...`, no TODO placeholders, no "rest unchanged".
3. Type hints, docstrings that state purpose, and short comments only where logic is non-obvious.
4. Tests (pytest) for each module, runnable with PowerShell commands.
5. Exact PowerShell commands to install, run and test.
6. "Expected output" describing what success looks like.
7. When I report an error, return only the changed files, in full, plus a one-line cause.

Reply now with only: "Charter understood. Ready for P1."

---

# P1: Foundation: config, schemas, logging, ID utilities, app skeleton

**Goal:** the shared base every other module imports.

**YOU PASTE:** the first ~40 lines of `feature_definition.json` (structure plus a few feature names), and the list of 132 feature names if short enough (else the file path layout only).

**Write these files:**

1. `backend/core/config.py`: Pydantic-settings loader reading `backend/configs/*.yaml` plus `.env`. A cached `get_settings()`. Settings groups: `paths` (all artifact locations from the charter, overridable), `risk_policy`, `investigation`, `graph_model`, `simulation`, `storage`, `api` (host/port/CORS for `http://localhost:5173`). Validate that referenced files exist only when `strict=True` (so tests can run without artifacts).
2. `backend/configs/app.yaml`, `risk_policy.yaml`, `investigation.yaml`, `graph_model.yaml` with sensible defaults. In `risk_policy.yaml`, define versioned policy objects: `alert_threshold`, `escalation_threshold`, `borderline_band: [low, high]`, `investigation_budget_per_10k`, `max_queue_depth`, and risk-band cut points (low/medium/high/critical). Use placeholders `0.95896` (alert, the validation best-F1 point) and `0.03219` (escalation, validation 80% recall) and comment that they are rank/budget-sensitive and the scores are uncalibrated.
3. `backend/core/schemas.py`: all Pydantic models listed in the charter. Requirements:
   - `TransactionEvent` contains the raw transaction fields only (no label), plus `event_id`, and optional `sim_metadata: dict` that is never used for features.
   - `GroundTruthRecord` (`transaction_id`, `is_laundering`, `scenario_id`, `scenario_type`, `scenario_version`, `role`, `difficulty`) is kept separate.
   - `DecisionRecord` has `transaction_id, risk_score, risk_band, decision, escalation_required, escalation_reason, model_version, feature_version, threshold_version, scoring_timestamp, latency_ms`.
   - `Subgraph` (nodes, edges, center accounts, `as_of_timestamp`, build params, coverage stats).
   - `GraphScore` (`score: float | None`, `availability` enum, `model_version`, `details`).
   - `InvestigationResult` per the fields in the project spec: `investigation_id, transaction_id, graph_risk_score, graph_availability, investigation_status, nodes_examined, edges_examined, lookback_window, max_hops, graph_statistics, suspicious_patterns, relevant_counterparties, model_version, processing_latency, error`.
   - `ShapExplanation` (base value, margin, top positive and negative contributors with feature name/value/shap, grouped-categorical view, metadata incl. disclaimer text).
   - `StreamMessage` as a discriminated union: `transaction_scored`, `alert_created`, `investigation_updated`, `simulation_status`, `error`.
4. `backend/core/ids.py`: `make_account_id(bank, account)` matching how `sender_id`/`receiver_id` were built (**make the separator a single constant and tell me where to check it against the preprocessing script**), transaction ID helpers, and a deterministic seed-derivation helper `derive_rng(seed, *labels)`.
5. `backend/core/logging.py`: structured JSON logging, a per-transaction trace-ID context var.
6. `backend/api/main.py`: minimal FastAPI app with `/health`, CORS, lifespan hook (empty for now), router registration stubs.
7. `requirements/backend.txt` (FastAPI, uvicorn[standard], pydantic v2, pydantic-settings, pyyaml, numpy, polars, pyarrow, xgboost, shap, scikit-learn, pytest, httpx, python-multipart as needed) and a `pyproject.toml` snippet if packages need to be importable (`pip install -e .`).
8. Tests: config loading, schema round-trip and validation, ID helper determinism.

---

# P2: Online feature engine (critical path) plus offline-online parity harness

**Goal:** reproduce the offline Phase 2A/2B F4 features exactly, event by event.

**YOU PASTE (mandatory; the bot cannot do this without it):**
- Full contents of `scripts/data/build_features_phase2a.py` and `scripts/data/build_features_phase2b.py`.
- Full `feature_definition.json`.
- `scripts/train/tabular_common.py` (the feature contract logic).
- Output of `polars` `.schema` for `modeling/splits/validation.parquet` (column names and dtypes).

**Write these files:**

1. `backend/services/features/feature_contract.py`: loads `feature_definition.json`; exposes ordered feature names, numeric/categorical split, categorical vocabularies (one-hot), the `±inf → NaN` rule, and `to_matrix(rows)` producing float32 arrays in exact model order with all-zero one-hot for unknown categories. This is the single source of truth for feature order.
2. `backend/services/features/rolling_windows.py`: efficient bounded rolling-window structures (deque-based, strictly `< t` semantics, windows up to 3 days) supporting count, sum, sum of squares, and unique-counterparty counts over windows 10m/1h/6h/24h/3d as required by the offline code. Document the exact boundary semantics (match `closed='left'`).
3. `backend/services/features/account_state.py`: per-account state (first-seen, last-event, lifetime counts, incoming/outgoing histories, rolling stats, counterparty sets with pair counts, flow statistics).
4. `backend/services/features/online_feature_engine.py`: `OnlineFeatureEngine` with:
   - `process_batch(events: list[TransactionEvent]) -> list[FeatureRow]`: events **must share one timestamp**; compute all features from pre-batch state, then update state. Reject out-of-order timestamps with a clear error.
   - Includes the helper statistics needed for z-scores/ratios even though they are not model inputs.
   - Reproduces NaN semantics for "no prior event" exactly as offline.
   - `group_by_timestamp(events)` utility for ordered streams.
5. `backend/services/features/snapshot.py`: save/load a pickled or compact snapshot of full engine state (so a warm-started state from the training period is created once and reused), plus `warm_start(engine, parquet_path, up_to_timestamp)` that replays events with **no scoring**, in chunks, with a progress bar.
6. `scripts/evaluate/feature_parity.py` (CLI via argparse): 
   - Select N target transactions (default 5,000) from the chosen split with a seeded sample, stratified to include all positives up to a cap.
   - Take the union of accounts involved, then replay **all** events touching those accounts in chronological order, batching equal timestamps, labels excluded.
   - Compare online vs offline features for the targets over the 132-feature contract **and** the underlying Phase 2A/2B columns they derive from.
   - Report per-feature: max abs diff, max rel diff, count of mismatches, NaN-pattern mismatches. Report the first mismatching transaction and feature. Write `results/reports/feature_parity_report.json` and a readable markdown summary.
   - Exit code non-zero if any mismatch exceeds tolerance (default exact for integer/flag features, 1e-6 relative for floats; configurable).
   - State in a comment why the account-union replay is valid (per-account features depend only on that account's own history) and what would invalidate it.
7. Tests: window boundary tests (event at exactly `t - window`, same-timestamp batch isolation), NaN-for-no-prior test, a tiny hand-computed example, determinism.

**Acceptance:** `feature_parity.py` reports zero unexplained mismatches on a validation sample. If any mismatch cannot be eliminated, the report must name the cause.

---

# P3: Model loading, XGBoost scorer, risk policy, IBM replay runner

**YOU PASTE:** `feature_definition.json` (if the bot's context lost it), `results/comparisons/threshold_analysis/operational_summary.json`, and the head of `validation_predictions.parquet` (5 rows).

**Write these files:**

1. `backend/services/inference/model_loader.py`: load `model.json` into an `xgboost.Booster` (and the sklearn wrapper if needed), record model version (name, file hash, XGBoost version), validate that the feature contract has 132 features, and return a `LoadedModel` object.
2. `backend/services/inference/xgboost_scorer.py`: `XGBoostScorer.score_rows(feature_rows) -> np.ndarray` (vectorised, supports batches of 1 for live and large batches for replay). Always build the matrix via `feature_contract.to_matrix`. Reject feature-name or order mismatch with a clear error. Measure and expose latency stats. Also expose `predict_margin()` (log-odds) for the SHAP service.
3. `backend/services/inference/risk_policy.py`: `RiskPolicy` (loads `risk_policy.yaml`) producing `DecisionRecord`. Implement:
   - Risk bands from configurable cut points.
   - `decision`: `alert` if score ≥ alert threshold, else `monitor`/`clear`.
   - Escalation triggers returning a **reason enum**: `HIGH_SCORE`, `BORDERLINE`, `GRAPH_RULE`, `NONE` (graph-rule input is injected later via a callable; default none).
   - A **token-bucket investigation budget** (max investigations per 10,000 events and max queue depth) so escalation cannot overwhelm the graph worker. Over-budget events are marked `escalation_required=True` but `investigation_deferred=True` with a reason. Nothing is silently dropped.
   - A `threshold_version` string derived from the policy config hash.
4. `scripts/evaluate/ibm_replay.py` (CLI): replay a chosen split (default **validation**) chronologically through `OnlineFeatureEngine` (with optional warm-start snapshot from P2) → scorer → policy. Support `--max-events`, `--warm-start`, `--out`. Write per-event decisions to parquet and a summary JSON: events/sec, mean and P95 per-event latency (feature + score), alert count, escalation count, investigation-budget deferrals, and PR-AUC/precision/recall/F1 and Precision@K against labels. **Compare against the stored `validation_predictions.parquet`** and report max abs diff in probability for matched `transaction_id`s (this is the end-to-end sanity check).
   - The test split requires an explicit `--allow-test` flag and prints a warning that thresholds must already be frozen.
5. `scripts/evaluate/policy_report.py`: from stored validation predictions, tabulate for several candidate policies (alert threshold at best-F1, escalation at 80/90/95% recall, top-K budgets): alert and escalation volume per 10,000 events, precision, recall, and projected investigations per hour at a given event rate. Output markdown and CSV in `results/comparisons/`.
6. Tests: policy decisions on synthetic scores, budget behaviour (burst of escalations), band boundaries.

---

# P4: SHAP service (tabular explainability)

**Goal:** fast local explanations plus global importance, with correct handling of one-hot groups.

**YOU PASTE:** `feature_definition.json` (needed for one-hot group names), and the xgboost and shap versions from `pip list`.

**Write these files:**

1. `backend/services/explainability/shap_service.py`: `ShapService` with:
   - `explain(feature_row) -> ShapExplanation` using the Booster's native `pred_contribs=True` (log-odds margin space; last column = bias). Verify in a test that contributions + bias equal `predict_margin` within tolerance, and that the `shap.TreeExplainer` result matches the native method on a sample (document the tolerance).
   - Top-k positive and negative contributors (default 10) with feature display name, raw value (NaN shown as "missing"), and SHAP value.
   - **Grouped view:** sum the one-hot columns of each categorical (`payment_format`, `payment_currency`, `receiving_currency`) into a single contribution labelled with the active category.
   - Human-readable feature labels via a configurable mapping (e.g. `sender_txn_count_1h` → "Sender transactions in last hour"). Generate the default mapping from feature-name patterns and allow overrides in `backend/configs/feature_labels.yaml`.
   - Metadata: model version, feature version, base value, predicted margin, score, and a fixed disclaimer ("Contributions describe the model's behaviour in log-odds space; they are associations in a synthetic dataset, not proof or causal explanations. Scores are uncalibrated.").
   - Batch method `explain_batch()` for replay/evaluation.
2. `backend/services/explainability/global_importance.py` and `scripts/evaluate/shap_global.py` (CLI): compute global mean |SHAP| over a sample of the validation split (default: all positives + a seeded random sample of negatives; document that this is a stratified sample, and give both the unweighted-sample importance and a prevalence-reweighted estimate). Output `results/models/xgboost_f4_v1/shap_global.json` plus a bar-chart PNG in `results/plots/`, plus a one-hot-grouped version. Compare with XGBoost's gain importance and report rank correlation.
3. `backend/services/explainability/shap_cache.py` (small LRU cache keyed by transaction ID) so repeated dashboard requests are cheap.
4. Tests: additivity, grouped sums equal ungrouped sums, NaN handling, label mapping fallbacks, latency check (print mean ms per explanation; no hard assert).

**Note for the bot:** SHAP is computed only on demand or for alerts/escalated events, never for every stream event.

---

# P5: Multi-agent transaction simulator

**YOU PASTE:** `polars` schema plus `.head(3)` of `accounts.parquet`; the head (first ~30 lines) of `HI-Small_Patterns.txt` so the scenario motifs are grounded in real pattern types; `.describe()`-style summary of `payment_format` and `payment_currency` value counts from the training split (a small table).

**Goal:** a reproducible stream of mixed legitimate and suspicious transactions, plus a separate ground-truth stream. Use lightweight policy classes, never an LLM.

**Write these files:**

1. `backend/services/simulation/account_registry.py`: builds a registry from `accounts.parquet` (account_id, bank_id, entity_type, role, created_time, behaviour_profile, known_counterparties). Bank is permanently tied to the account. Selects a **small active population** (configurable, default ~2,000 accounts) with assigned roles (individual, business, merchant, salary_payer, regular_counterparty, mule/layer participants for scenarios). Supports generating additional **synthetic new accounts** (flagged `origin="synthetic"`) for cold-start tests. Seeded.
2. `backend/services/simulation/agents.py`: strategy/policy classes (`IndividualPolicy`, `BusinessPolicy`, `MerchantPolicy`, `SalaryPayerPolicy`, `RegularCounterpartyPolicy`) generating transaction intents from account state. Amounts are sampled from role-conditioned empirical distributions (provide a small lookup config derived from IBM summary stats; loaded from `backend/configs/amount_profiles.yaml`). Payment formats and currencies are sampled from empirical IBM distributions, **never uniformly**, and are consistent per account/pair.
3. Legitimate scenario coverage (must all exist): salary/recurring income, household transfers, merchant payments, B2B payments, recurring counterparties, **legitimate payment to a new recipient**, **high-volume business account**. The last two deliberately resemble suspicious features in isolation (new counterparty, high count) for false-positive evaluation.
4. `backend/services/simulation/scenarios.py` plus `backend/configs/scenarios/*.yaml`: a declarative, versioned scenario library with fields `scenario_id, scenario_version, scenario_type, participating_roles, required_account_population, generation_policy, graph_motif, event_schedule, duration, ground_truth_labels, difficulty_level, split` (`dev` or `heldout`). Implement **rapid pass-through, circular, mule network** first, and **layering and dormant-account reactivation** as additional scenarios. Include 2+ parameter variants for each of at least 3 scenario types tagged `heldout` (for experiment E4). Add a loader that rejects any scenario missing required fields. These are synthetic detector-test motifs only. Do not include operational evasion advice.
5. `backend/services/simulation/transaction_generator.py`: turns intents into `TransactionEvent`s and `GroundTruthRecord`s. Positive-label rule: **only the designated laundering edges of a scenario are positive** (mirroring IBM labelling); other scenario-context edges are labelled negative with `scenario_id` kept for analysis. Document this rule at the top of the file.
6. `backend/services/simulation/event_scheduler.py`: a discrete-event scheduler using a priority queue and an event clock with plausible inter-arrival (e.g. non-homogeneous Poisson with daily seasonality), handling many agents without per-agent objects. Equal timestamps are possible and must be emitted as a group in order.
7. `backend/services/simulation/controller.py`: `SimulationController` with `start, pause, resume, stop, reset, set_speed, run_n_events, select_scenarios`, running in a background asyncio task and emitting events to an output `asyncio.Queue` (events and ground truth are on **separate** queues). Exposes `status()` (state, events emitted, active scenarios, seed, config hash).
8. Output files for reproducibility: `results/metrics/simulations/<run_id>/{config.json, events.parquet, ground_truth.parquet}` written on demand by a helper.
9. `scripts/utilities/run_simulation_headless.py`: generates N events with a seed and writes the parquet outputs; running it twice with the same seed must produce byte-identical event files.
10. Tests: seeded reproducibility, bank consistency per account, no uniform-random bank assignment, scenario schema validation, ground-truth separation (assert the `TransactionEvent` dump contains no label field), pass-through scenario produces the expected motif in the output.

---

# P6: Temporal graph store, graph builder, analytics, rules, escalation, async investigation

**YOU PASTE:** the `Subgraph`/`InvestigationResult` schema block from P1 (if the bot's context has drifted), and a note on whether the 35 edge-feature names used by `graph_v1` should be mirrored here (paste the list of the 35 edge feature names and 12 node feature names if available; otherwise the bot defines a documented subset and the mapping is finalised in P7).

**Write these files:**

1. `backend/services/investigation/temporal_graph_store.py`: an in-memory, append-only temporal edge store. Per account: time-sorted incoming and outgoing edge lists with `bisect` for time cut-offs. API: `add_events(events)` (called **after** a batch is scored, preserving strict prior-time semantics), `edges_before(account, t, lookback, limit)`, `neighbors_before(...)`. Bounded retention (configurable max age) with eviction. `bulk_load(parquet, up_to)` for warm-start from IBM history.
2. `backend/services/investigation/graph_builder.py`: `build_subgraph(sender, receiver, t, params)`. Implements BFS expansion from both accounts, using **only edges with timestamp < t**, with `lookback_window`, `max_hops` (default 2), `max_nodes`, `max_edges`, per-node neighbour caps (most recent first, then largest amount), deterministic ordering for reproducibility, and `coverage` metrics (fraction of expansion truncated, history depth). The candidate transaction is **never** included. Return a `Subgraph` with node features and edge features (amount, log amount, time delta to t, payment format, currencies, cross-bank flag).
3. `backend/services/investigation/graph_analytics.py`: interpretable statistics for the subgraph and center accounts: in/out degree, unique counterparties, weakly connected component size, local transaction volume, repeated-interaction counts, activity concentration (e.g. top-counterparty share), flow-through ratio per center account, and **cycle detection** (length 2-4 through the center accounts, bounded). All output is serialisable and described as "investigation signals, not proof".
4. `backend/services/investigation/graph_rules.py`: lightweight rules usable as an escalation trigger **without** building the full subgraph (cheap checks from the store): fan-in/fan-out burst over a short window, rapid pass-through (inflow followed by outflow of similar amount within minutes), dormant-account reactivation, repeated 2-cycle. Each rule returns `RuleHit(rule_id, evidence)`. Rules are configured in `investigation.yaml` and independently toggleable. Provide `GraphRuleTrigger.check(event) -> list[RuleHit]`, wired into `RiskPolicy` as the `GRAPH_RULE` trigger.
5. `backend/services/investigation/escalation.py`: combines the first-stage `DecisionRecord`, rule hits and budget state into a final escalation decision with reason(s) logged.
6. `backend/services/investigation/job_queue.py`: asynchronous investigation queue (asyncio, bounded, priority by score, a configurable number of worker tasks; graph work runs in a thread executor). Handles failures: retries, timeouts, and a stored error status. **A job failure must never alter the original decision.** Exposes queue depth and completion stats.
7. `backend/services/investigation/investigation_service.py`: `InvestigationService.run(event, decision, feature_row) -> InvestigationResult`:
   - Build subgraph → compute analytics → call `GraphScorer` (from P7; for now accept any object with the interface, defaulting to `NullScorer`) → assemble `InvestigationResult`.
   - **Cold-start handling:** set `graph_available`, `graph_partially_available`, or `graph_unavailable_cold_start` from configurable thresholds on center-account history. In cold start, do not fabricate connections, do not raise the risk score, and return the first-stage score context.
   - Record nodes/edges examined, processing latency, and any errors.
8. Tests: no future edges (inject an event with a later timestamp and assert it never appears), the candidate edge is excluded, same-timestamp events exclude each other, caps respected, cold-start status, a planted 3-cycle is found, rule hits on planted patterns, queue failure isolation.

---

# P7: Pluggable GraphSAGE model interface, Colab export, and loader (slot-in mechanism)

**Goal:** the graph model is a replaceable artifact. The current weak Colab model works today. The improved model, trained later, drops in by copying a bundle folder and changing one config value, with **no code changes**.

**YOU PASTE (mandatory):** the **Colab notebook code that defines and trains the current GraphSAGE model** (model class, how embeddings are computed, edge classifier, how node/edge features were built and normalised, the neighbour sampler settings, and how the model is saved now). Plus the `graph_v1` node and edge feature name lists, and the `export_gnn_dataset.py` script.

**Design requirements for the bot:**

- **Bundle format** (a folder `research/models/graph/<model_name>/<version>/`):
  - `weights.pt` (a `state_dict`; not a pickled model object).
  - `manifest.json` containing: `model_name, version, architecture` (a registry key, e.g. `graphsage_edge_v1`), `framework_versions`, `num_layers`, `hidden_dims`, `embedding_dim`, `classifier_dims`, `dropout`, **`node_feature_names` (ordered)**, **`edge_feature_names` (ordered)**, `normalization` (per-feature mean/std or min/max, with names), `categorical_encodings`, `neighbor_fanouts`, `inference_mode` (`local_subgraph`), `classifier_input_composition` (e.g. `[src_emb, dst_emb, edge_features]`), `training` info (graph version, split, seed, commit, date), `reported_metrics`, and `parity_fixture` path.
  - `parity_fixture.npz/json`: a small set of saved inputs (subgraphs encoded as tensors) and the expected output scores produced in Colab, used to verify the local loader reproduces Colab outputs.
- **Pure-PyTorch SAGE layer** (`sage_layers.py`) with parameter names that match PyG `SAGEConv` (`lin_l.weight`, `lin_l.bias`, `lin_r.weight`; mean aggregation) so state dicts trained with `torch_geometric` load locally **without installing torch-geometric**. Add a test that loads a PyG-style state dict.
- **Architecture registry:** `registry.py` maps `architecture` strings to builder classes. Adding a new architecture (e.g. a temporal GNN or GAT) means adding one file and one registry entry.
- **Node features at inference** are computed from the bounded subgraph by `node_features.py`, keyed by `node_feature_names` from the manifest and normalised using the manifest stats. Missing or unavailable features are handled per the manifest (explicit policy: `zero`, `mean`, or `error`). **A feature-name mismatch between the manifest and what the builder can supply must raise a clear error listing missing names**, never silently reorder or fill.
- `GraphScorer` protocol in `base.py`: `score(subgraph, target_event, context) -> GraphScore`, `model_info() -> dict`, `healthcheck()`.
- `null_scorer.py`: returns `GraphScore(score=None, availability=..., details={"reason": "graph model disabled"})`. This is the safe default.
- `loader.py`: `load_graph_model(manifest_path_or_dir)` validates the manifest schema, checks `framework_versions` compatibility (warn on mismatch), builds the architecture, loads weights (strict), runs the **parity fixture** and fails loudly if outputs differ beyond tolerance, and returns a `GraphScorer`. `graph_model.yaml` selects the active bundle (`active_model: <name>/<version>` or `null`). Hot-reload endpoint support (a `reload()` method) for P8.
- **Colab export script** (give as a ready-to-paste Colab cell block plus a standalone `scripts/graph/export_bundle_colab.py`): given the trained model, produces the whole bundle folder as a zip, including manifest, scaler stats, and parity fixture generated from a handful of validation edges, with the **exact same local-subgraph inference procedure** (cut-off at `t`, fanouts, features) used locally. Include a `scripts/graph/validate_bundle.py` that I can run locally on a downloaded bundle and that prints a pass/fail table (manifest valid, features resolvable, weights load, parity passes, latency per scorer call on CPU).
- State clearly in the docs/comments the **known mismatch risk**: the model was trained on static train-only node aggregates over a full training topology, whereas online inference builds node features from a bounded subgraph. The manifest must carry a field `node_feature_mode` (`train_static_aggregates` or `subgraph_local`), and the loader must refuse a `train_static_aggregates` bundle unless a documented `static_node_feature_table` path is supplied in the manifest. Implement this refuse/allow logic. Tell me which mode the current Colab model is in, based on the code I pasted, and what must change in the *next* training run so that the exported model uses `subgraph_local` features (a short bullet list I can take into the later training prompt).
- Wrap the **current** Colab model into a bundle so the pipeline runs end to end today (with its weak scores), clearly labelled `version: v0-baseline-weak` in the manifest.
- Tests: manifest validation failures, ordering mismatch raising, parity pass/fail, `NullScorer` fallback when `active_model: null`, registry unknown-architecture error.

---

# P8: Pipeline orchestrator, storage, report builder, FastAPI routes, WebSocket

**YOU PASTE:** nothing mandatory; if schemas changed in P2-P7, paste the latest `schemas.py`.

**Write these files:**

1. `backend/services/storage/db.py` (+ `transaction_store.py`, `alert_store.py`, `investigation_store.py`): SQLite via the standard `sqlite3` or SQLAlchemy (choose the simpler); schema creation, WAL mode, indexed by `transaction_id`, `alert_id`, `investigation_id`, timestamp. Stores: processed transactions (event + features snapshot hash + decision), alerts, investigations, subgraphs (JSON), explanations (JSON), reports (JSON). Ground truth is stored in a **separate table** used only by the evaluation endpoints. Retention/cap options to keep the demo database small.
2. `backend/services/pipeline/event_bus.py`: an in-process pub/sub that fan-outs `StreamMessage`s to WebSocket clients with bounded per-client queues and drop-oldest backpressure (never blocking the pipeline).
3. `backend/services/pipeline/orchestrator.py`: ties it all together:
   - Input: events from the simulator queue or the IBM replay source.
   - `group_by_timestamp` → `OnlineFeatureEngine.process_batch` → `XGBoostScorer` → `RiskPolicy` → persist → publish `transaction_scored` (this happens **before and independently of** any graph work).
   - After scoring, update the temporal graph store (post-batch), then evaluate escalation (budget, rules) → create an alert if applicable → enqueue an investigation job → on completion run SHAP + report builder → persist → publish `investigation_updated`.
   - Strict ordering guarantees: feature state mutations are serialised (single consumer). Investigation workers only read from the graph store using time cut-offs.
   - Error isolation, per-stage latency metrics, graceful start/stop, and a `warm_start` option (load the snapshot from P2).
   - A mode switch: `source = simulator | ibm_replay(split, speed)`.
4. `backend/services/explainability/graph_evidence.py` and `report_builder.py`: build the structured `InvestigationReport` containing transaction summary; XGBoost score, band and decision; escalation reason(s); SHAP explanation (top contributors, grouped view); graph availability and coverage; accounts examined; relevant historical transactions; graph statistics and rule hits; GraphSAGE score (nullable) with model version; cold-start status; combined narrative summary generated from templates (plain, factual, hedged language, no accusations); model, feature and threshold versions; timestamps. **XGBoost and graph scores are reported side by side; no hidden blended "final score".** Add an explicit `limitations` list in every report.
5. API routes (`backend/api/routes/*.py`), all returning Pydantic models:
   - `GET /health`, `GET /models/info` (XGBoost info, active graph bundle info or `null`), `POST /models/graph/reload`.
   - Simulation: `POST /simulation/start|pause|resume|stop|reset`, `GET /simulation/status`, `GET /simulation/scenarios`, `POST /simulation/config`. Source selection (simulator vs IBM replay).
   - `GET /transactions` (filters, pagination, sort), `GET /transactions/{id}`.
   - `GET /alerts`, `GET /alerts/{id}`.
   - `GET /investigations`, `GET /investigations/{id}`, `GET /investigations/{id}/graph`, `GET /investigations/{id}/explanation`, `GET /investigations/{id}/report`.
   - `GET /metrics/overview` (processed count, alert count by band, investigations triggered/completed/failed, queue depth, mean and P95 scoring latency, mean and P95 investigation latency, graph invocation rate, active scenarios).
   - `GET /evaluation/summary` (uses the separate ground-truth table to compute live precision/recall/alerts-per-10k for the current run; clearly separated from detection paths and labelled as simulation-only).
   - `GET /explain/global` (serves the saved global SHAP importance).
   - `WS /stream`: pushes `StreamMessage`s (types from P1).
6. `backend/api/main.py` lifespan: construct services once, wire them, start and stop the orchestrator cleanly.
7. `docs/api_contract.md` (endpoint table with example JSON) and `frontend/src/api/types.ts` (TypeScript types mirroring the Pydantic schemas) so the frontend prompts use an exact contract.
8. Tests (httpx + pytest-asyncio): end-to-end on a 2,000-event headless simulation; assert every event has a decision record; assert investigation failure injection does not affect the stored decision; WebSocket receives messages; ground truth never appears in non-evaluation endpoints.

PowerShell: include the `uvicorn backend.api.main:app --reload --port 8000` command.

---

# P9a: Frontend app shell, API client, overview, live stream, alerts

**YOU PASTE:** `docs/api_contract.md` and `frontend/src/api/types.ts` from P8.

**Stack (fixed):** React 18 + Vite + TypeScript, Tailwind CSS, React Router, TanStack Query, native WebSocket, Recharts for charts. (Cytoscape.js is added in P9b.) I will install Node myself. Give the exact PowerShell scaffold commands (`npm create vite@latest ...`, `npm i ...`, Tailwind setup) and `npm run dev` (port 5173).

**Write these files:**

1. Project config (`package.json` deps list, `vite.config.ts` with a dev proxy to `http://localhost:8000` for REST and WS, `tailwind.config.js`, `tsconfig`), `src/main.tsx`, `src/App.tsx` with routes: `/` Overview, `/stream` Live stream, `/alerts` Alerts, `/investigations/:id` (stub for P9b), `/simulation` (stub for P9b), `/evaluation` (stub for P9b).
2. `src/api/client.ts` (typed fetch wrapper with error handling), `src/api/ws.ts` (a reconnecting WebSocket hook with exponential backoff, connection-status indicator, and message-type dispatch), React Query hooks for each endpoint.
3. `src/state/streamStore.ts` (a lightweight store, e.g. Zustand, or React context) holding a **bounded ring buffer** of the latest N (default 500) scored transactions and alerts, updated from the WebSocket; pause/resume of the live view without dropping the connection.
4. Layout: sidebar navigation, header with simulation status chip, WS connection indicator, and a persistent banner "Synthetic data · simulation sandbox · not for real-world decisions". Dark/light theme toggle. Responsive, accessible, keyboard-focusable tables.
5. **Overview page:** KPI cards (processed, alerts, high-risk alerts, investigations triggered/completed/failed, queue depth, mean and P95 scoring latency, investigation latency, graph invocation rate), a risk-band distribution chart, a throughput/alerts-over-time chart, and a recent alerts list.
6. **Live stream page:** virtualised table (e.g. `@tanstack/react-virtual`) with timestamp, transaction ID, sender, receiver, amount, currency, risk score (with a colour-coded bar), risk band badge, investigation status badge; filters (band, status, min score), text search, row click opens a side drawer with the transaction detail and a link to its investigation if one exists.
7. **Alerts page:** server-paginated table with sorting and filters (band, escalation reason, investigation status, date), alert detail drawer, and badge components for escalation reason (`HIGH_SCORE`, `BORDERLINE`, `GRAPH_RULE`) with tooltips explaining each. Show clearly when an investigation was **deferred by budget**.
8. Shared components: `RiskBadge`, `StatusBadge`, `ScoreBar`, `KpiCard`, `EmptyState`, `ErrorState`, `Skeleton`, formatting helpers (currency, timestamps in UTC with a relative hint).
9. `README` snippet listing the run steps.

Include a `frontend/src/mock/` mock-data mode (toggle by an env var) so the UI can be tested before the backend runs.

---

# P9b: Frontend investigation detail, SHAP view, graph visualisation, simulation control, evaluation view

**YOU PASTE:** the current `types.ts`, plus the actual JSON from `GET /investigations/{id}`, `/graph`, `/explanation`, `/report` for one real investigation, once the backend runs (pasting real responses avoids contract drift).

**Add dependencies:** `cytoscape`, `cytoscape-dagre` (or `cytoscape-fcose`), and `react-cytoscapejs`.

**Write these files:**

1. **Investigation detail page** (`/investigations/:id`), sectioned layout:
   - Header: transaction summary, status timeline (received → scored → escalated → graph built → graph scored → explained), and the escalation reason(s).
   - **Initial XGBoost decision card:** score, band, decision, thresholds and policy version, with a note that the score is uncalibrated and was returned before graph processing (show the timestamps proving it).
   - **SHAP panel (`ShapChart.tsx`):** horizontal waterfall-style bars of top positive (red) and negative (blue) contributors with feature label, feature value and SHAP value, a toggle for **grouped vs raw one-hot features**, tooltips with the raw feature name, the base value and the final margin, and the disclaimer text from the API always visible. A link to the global importance chart (`/explain/global`).
   - **Graph panel (`GraphView.tsx`, Cytoscape):** nodes coloured by role (sender, receiver, neighbour), sized by degree or volume, edges with width scaled to log amount and direction arrows, a hover tooltip (amount, time, format, currency), click-to-inspect side panel, **highlighting of detected patterns** (cycle edges, rule-hit paths), a time slider or play button to reveal edges chronologically, layout selector, zoom and fit controls, a legend, truncation warnings when caps were hit, and export as PNG. Handle graphs of up to a few hundred nodes smoothly.
   - **Graph evidence panel:** graph availability badge (available / partial / **cold-start** with explanatory text), coverage figures, statistics table, rule hits, relevant counterparties table, **GraphSAGE score box** showing the score, model name/version, or "graph model not active". Show XGBoost and graph scores **side by side with no blended score**.
   - Report panel: combined narrative, limitations list, versions, a "Download report (JSON)" button, and a print-friendly view.
2. **Simulation control page:** scenario picker (legitimate and suspicious, with descriptions and difficulty, and `dev`/`heldout` tags), source selector (simulator vs IBM replay split), seed, number of events or continuous mode, a speed slider, start/pause/resume/stop/reset buttons wired to the API with optimistic status and error toasts, active scenario list, and live counters.
3. **Evaluation view** (`/evaluation`): clearly labelled **"Simulation ground truth, evaluation only"**. Shows live confusion counts, precision, recall, alerts per 10k, scenario detection table (scenario, detected?, time to detection), and a comparison of XGBoost-only versus XGBoost+graph rules versus +GraphSAGE where the API provides it. Include a button to export the current run's metrics as CSV.
4. **Demo helpers:** a "Demo mode" panel with one-click presets: (a) mixed stream, (b) legitimate new-recipient case, (c) cold-start account case, (d) pass-through scenario, each of which configures the simulation through the API. Include a guided-tour checklist matching the final demonstration sequence.
5. Components tests (Vitest + React Testing Library) for `ShapChart` (grouping toggle), `GraphView` (renders nodes/edges from fixture, highlights cycle), and the cold-start badge logic.

---

# P10: Experiment harness (Tracks A and B, E1-E5)

**YOU PASTE:** the actual output summaries from P3 (`ibm_replay` summary JSON), `policy_report` CSV, and a sample simulation `config.json` from P5.

**Goal:** config-driven, reproducible experiments with figures and tables for the report. Never mix tracks.

**Write these files:**

1. `research/configs/experiments/*.yaml`: one config per experiment (seed, source, scenarios, policy, graph model, warm-start, sample sizes).
2. `scripts/evaluate/run_experiment.py`: runs an experiment config headlessly through the same orchestrator code path (no UI), writing to `results/metrics/<experiment_id>/` with the config, seed, Git revision (if available), model/feature/threshold versions, and artifact hashes.
3. **Track A (IBM replay):** validation (policy selection), then test (**single run with frozen policy, requires `--allow-test`**), then tail_stress (separate stress report). Metrics: PR-AUC, ROC-AUC, Precision/Recall@K, precision, recall, F1, FPR, FP per 10,000, alert volume, escalation/investigation rate, deferral rate, mean and P95 latency by stage, with **bootstrap 95% confidence intervals**.
4. **Track B (simulator):** E1 legitimate-only false positives (including new-recipient and high-volume-business subsets); E2 known dev scenarios (precision, recall, F1, scenario detection rate, time to detection); E3 cold-start accounts (detection, FPR, graph-availability breakdown); E4 held-out scenarios (evaluated only with a frozen configuration; the script refuses to tune on `heldout` scenarios and logs that); E5 incremental graph value across four configurations: XGBoost-only, + graph rules, + GraphSAGE, + both (recall gained among XGBoost misses, false positives added, invocation rate, latency, extra cost, and an analyst-context checklist score where the evidence was available).
5. `scripts/evaluate/make_report.py`: assembles tables (CSV and markdown) and plots (PR curves, FP-per-10k versus recall, latency distributions, alert-volume funnels, graph-availability breakdown, E5 ablation bars with CIs) into `results/reports/evaluation_report.md` and `results/plots/`. The report auto-includes a **Limitations** section (synthetic data, simulator fidelity, uncalibrated scores, small positive counts, single dataset, GNN not outperforming XGBoost unless the data say so).
6. Tests for metric functions (known inputs), bootstrap CI sanity, the heldout-guard behaviour.

---

# P11: Documentation and demonstration pack

Write, from the actual implemented modules (ask me to paste anything you need):

1. `docs/architecture.md` with Mermaid diagrams: system architecture, data flow, escalation sequence (showing the first-stage response before graph work), and the graph-model plug-in lifecycle (Colab → bundle → validate → activate).
2. `docs/feature_engineering.md` (offline vs online parity method and results), `docs/model_methodology.md` (XGBoost, thresholds, uncalibrated scores, SHAP caveats), `docs/simulator_methodology.md` (agents, scenarios, labelling rule, held-out policy), `docs/graph_investigation.md` (subgraph construction, temporal rules, cold-start, GNN interface), `docs/limitations.md`.
3. `docs/setup.md`: PowerShell setup from zero (venv, requirements, running backend, running frontend, validating a model bundle, running headless experiments).
4. `docs/demo_script.md`: a step-by-step walk-through of the final demonstration (live stream, escalation, async investigation, SHAP, graph, legitimate new-recipient case, cold-start case, ground-truth view, results and latency), with the exact seeds and presets that reproduce each moment.

---

# LATER (separate prompt, not part of this series): GraphSAGE improvement and retraining

Placeholder. When you are ready, a dedicated prompt will cover: retraining on Colab with `subgraph_local` node features; temporal-correct neighbour sampling; class-imbalance handling; stronger edge features; validation-based model selection; and exporting the result with `scripts/graph/export_bundle_colab.py` from P7. The new model drops in by placing its bundle under `research/models/graph/<name>/<version>/`, running `scripts/graph/validate_bundle.py`, and setting `active_model` in `backend/configs/graph_model.yaml` (or calling `POST /models/graph/reload`). No application code changes should be required.
