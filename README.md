# FinCrime Intelligence

### Modular AI Platform for Financial Crime Detection, Risk Intelligence, and Automated Investigation

FinCrime Intelligence is a research-first financial crime analysis platform designed to experiment with and evaluate modern approaches for detecting suspicious financial activity.

The project begins with a domain-neutral financial fraud architecture and is designed to later specialize into areas such as:

- Anti-Money Laundering (AML)
- Sanctions Evasion Detection
- Suspicious Transaction Monitoring
- Mule Account Detection
- Financial Network Analysis
- Entity Risk Intelligence
- Automated Financial Crime Investigation

The objective is not to build a single fraud classifier.

Instead, the project provides a modular experimentation environment where traditional Machine Learning, behavioural analytics, graph intelligence, Graph Neural Networks, anomaly detection, explainability, and investigation tooling can be independently evaluated and progressively combined.

---

# 1. Project Objective

Traditional financial crime monitoring systems frequently rely on static rules and transaction-level thresholds.

Examples include:

```text
Transaction amount > threshold
Transaction from high-risk country
Multiple transfers within fixed interval
Known sanctioned entity detected
```

Although useful, such approaches often struggle when suspicious behaviour emerges through:

- multiple seemingly legitimate transactions;
- networks of connected accounts;
- rapid movement of funds;
- layering across several intermediaries;
- circular fund flows;
- mule account networks;
- changing behavioural patterns;
- indirect connections to high-risk entities.

FinCrime Intelligence therefore investigates financial crime through multiple complementary representations:

```text
                         FINANCIAL ACTIVITY

                                │
              ┌─────────────────┼─────────────────┐
              │                 │                 │
              ▼                 ▼                 ▼
          TABULAR           TEMPORAL           GRAPH
          FEATURES          BEHAVIOUR          STRUCTURE
              │                 │                 │
              ▼                 ▼                 ▼
        Classical ML       Sequence ML       Graph Analytics
                                                / GNN
              │                 │                 │
              └─────────────────┼─────────────────┘
                                ▼
                           RISK ENGINE
                                │
                                ▼
                          EXPLAINABILITY
                                │
                                ▼
                        INVESTIGATION CASE
                                │
                                ▼
                         ANALYST INTERFACE
```

The system is intentionally modular so that different algorithms, datasets, features, and financial crime hypotheses can be tested without redesigning the entire platform.

---

# 2. Research Philosophy

The project follows one central principle:

> Build the platform first. Allow the experiments to determine the specialization.

The project is currently not restricted to a single financial crime category.

The initial baseline focuses on transaction monitoring because publicly available AML and fraud datasets provide useful transaction graphs and known suspicious behaviour.

Once the baseline platform is established, experiments will determine whether the strongest research contribution lies in:

```text
Money Laundering Detection
        │
        ├── Transaction behaviour
        ├── Layering
        ├── Structuring
        ├── Circular transfers
        ├── Mule networks
        └── Temporal transaction graphs

or

Sanctions Evasion Detection
        │
        ├── Entity resolution
        ├── Beneficial ownership
        ├── Indirect relationships
        ├── Knowledge graphs
        ├── Fuzzy identity matching
        └── Sanctioned-entity proximity
```

This prevents the research from forcing novelty before experimental evidence exists.

---

# 3. Core Design Principles

## Modular Architecture

Each component has a defined input and output.

A model, database, or graph algorithm should therefore be replaceable without rewriting unrelated components.

---

## Research and Production Separation

Experimental code and stable application code are intentionally separated.

```text
research/
```

answers:

> What should we experiment with?

while:

```text
backend/
```

answers:

> What has survived experimentation and can become part of the platform?

---

## Privacy by Design

Models should operate primarily on pseudonymized identifiers and derived features rather than directly consuming personally identifiable information.

The long-term architecture separates:

```text
Identity Information
        │
        ▼
Secure Identity Layer

        SEPARATE FROM

Internal Entity ID
Transaction Features
Graph Features
Risk Intelligence
```

---

## Explainability

The goal is not merely:

```text
Suspicious = True
```

The system should eventually provide:

```text
Risk Score: 91 / 100

Key Evidence
----------------------------------
High transaction velocity
Low fund-retention time
Multiple new counterparties
Participation in transaction cycle
Abnormal amount relative to history
Connection to suspicious entity
```

---

## Reproducible Research

Every meaningful experiment should record:

- dataset;
- feature set;
- model;
- hyperparameters;
- random seed;
- data split;
- metrics;
- runtime;
- Git commit;
- experiment identifier.

Research findings should therefore be reproducible rather than dependent on undocumented notebook runs.

---

# 4. System Architecture

The planned system consists of several independent layers.

```text
                         DATA SOURCES
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
    Transactions             KYC            External Intel
                                            Sanctions etc.
          │                   │                   │
          └───────────────────┼───────────────────┘
                              ▼
                    DATA INGESTION LAYER
                              │
                              ▼
                   VALIDATION & CLEANING
                              │
                              ▼
                  PRIVACY / ENTITY LAYER
                              │
                              ▼
                    CANONICAL DATA MODEL
                              │
             ┌────────────────┼────────────────┐
             │                │                │
             ▼                ▼                ▼
      Transaction        Behavioural         Graph
        Features           Features         Builder
             │                │                │
             ▼                ▼                ▼
          Tabular         Temporal          Network
            ML             Models           Analytics
                                               │
                                               ▼
                                              GNN
             │                │                │
             └────────────────┼────────────────┘
                              ▼
                        RISK FUSION
                              │
                              ▼
                       EXPLAINABILITY
                              │
                              ▼
                            ALERT
                              │
                              ▼
                             CASE
                              │
                              ▼
                   INVESTIGATION SERVICES
                              │
                              ▼
                     ANALYST DASHBOARD
```

Not all modules are implemented during the initial baseline.

The architecture allows them to be introduced progressively.

---

# 5. Technology Stack

## Core Language

### Python 3.13

Python 3.13 is used for:

- data engineering;
- machine learning;
- graph processing;
- experimentation;
- feature engineering;
- backend services;
- risk intelligence.

Project requirement:

```text
Python >=3.13,<3.14
```

---

# 6. Data Engineering

Primary libraries:

```text
Pandas
NumPy
PyArrow
Polars
```

### Pandas

Primary compatibility layer for tabular financial data.

### NumPy

Numerical operations.

### PyArrow

Used primarily for Apache Parquet support.

### Polars

Available for experimentation with high-volume transaction processing.

---

# 7. Data Storage Format

Raw datasets may originate as:

```text
CSV
JSON
API responses
database exports
```

Internally, processed financial data should primarily use:

### Apache Parquet

Pipeline:

```text
RAW CSV
   │
   ▼
Dataset Adapter
   │
   ▼
Canonical Schema
   │
   ▼
Parquet
   │
   ▼
Feature Engineering / ML / Graph Construction
```

Parquet provides:

- compression;
- typed columns;
- faster analytical reads;
- interoperability with modern data systems.

---

# 8. Machine Learning Stack

Initial models:

```text
Logistic Regression
Random Forest
XGBoost
```

Libraries:

```text
scikit-learn
XGBoost
imbalanced-learn
```

These models establish the baseline against which more complex techniques are compared.

Example research progression:

```text
Logistic Regression
        ↓
Random Forest
        ↓
XGBoost
        ↓
XGBoost + Behavioural Features
        ↓
XGBoost + Graph Features
        ↓
Graph Neural Networks
        ↓
Temporal / Hybrid Models
```

Complexity must justify itself through experimental performance.

---

# 9. Graph Intelligence

Graph processing is divided into two levels.

## Classical Graph Analytics

Initial library:

```text
NetworkX
```

Planned graph features include:

```text
In-degree
Out-degree
Weighted degree
PageRank
Betweenness centrality
Community membership
Fan-in behaviour
Fan-out behaviour
Transaction cycles
Graph proximity
Multi-hop relationships
```

---

## Graph Neural Networks

Planned framework:

```text
PyTorch
PyTorch Geometric
```

Candidate models:

```text
GCN
GraphSAGE
GAT
Temporal GNN variants
```

Graph Neural Networks are treated as experimental research components rather than hard dependencies for the baseline platform.

---

# 10. Graph Database

Planned application-level graph database:

### Neo4j

Neo4j will eventually support:

- investigator graph exploration;
- entity relationship queries;
- multi-hop traversal;
- transaction path discovery;
- sanctions relationship analysis;
- network visualization.

For early research:

```text
Parquet + NetworkX
```

is sufficient.

Neo4j should not become a dependency for basic model experimentation.

---

# 11. Relational Database

Planned relational database:

### PostgreSQL

Used for application data such as:

```text
Transactions
Entities
Alerts
Cases
Users
Model metadata
Audit events
Investigation status
```

The design intentionally separates transactional application storage from graph relationship storage.

---

# 12. Backend

Backend framework:

### FastAPI

Used for:

- model serving;
- transaction scoring;
- graph queries;
- case management;
- analyst requests;
- investigation services.

Planned architecture:

```text
Analyst Frontend
       │
       ▼
    FastAPI
       │
       ├── Transaction Services
       ├── Feature Services
       ├── Graph Services
       ├── Model Services
       ├── Risk Services
       ├── Explanation Services
       └── Investigation Services
```

The frontend never directly imports or executes ML models.

---

# 13. Analyst Frontend

The analyst-facing application is kept separate from the Python backend.

Planned stack:

```text
React
TypeScript
Vite
Tailwind CSS
shadcn/ui
Cytoscape.js
Recharts / Plotly
```

Recommended Node environment:

```text
Node.js 24 LTS
```

---

## Planned Analyst Interfaces

### Dashboard

High-level system overview.

```text
Transactions Processed
Active Alerts
Critical Alerts
Open Investigations
Risk Distribution
Recent Activity
```

---

### Alert Queue

Allows investigators to filter and prioritize suspicious activity.

---

### Case Investigation

Primary investigation workspace combining:

```text
Risk Score
Transaction Graph
Transaction Timeline
Model Evidence
Graph Evidence
Entity Information
Explanation
Investigation Notes
```

---

### Entity Profile

Provides a consolidated view of an account, individual, organization, wallet, or other financial entity.

---

### Transaction Network

Interactive visualization using Cytoscape.js.

---

### Research / Model Metrics

Displays performance of deployed or experimental models.

---

# 14. Explainable AI

Initial explainability framework:

### SHAP

Used primarily for:

```text
XGBoost
Random Forest
Tabular classifiers
```

Planned graph explainability may later include:

```text
GNNExplainer
Captum
Graph-derived evidence
Path explanations
```

The goal is to eventually explain both:

```text
WHY did the model classify this activity as risky?

and

WHAT relationships contributed to the decision?
```

---

# 15. Risk Intelligence

The final system is not expected to depend on one classifier.

Instead, multiple signals may eventually contribute to the risk score.

```text
ML Risk
   │
Graph Risk
   │
Behavioural Risk
   │
Anomaly Risk
   │
Rule-Based Risk
   │
Sanctions Risk
   │
   ▼
RISK FUSION ENGINE
   │
   ▼
Final Risk Score
```

Risk fusion itself is an experimental research opportunity.

Potential approaches include:

```text
Fixed weighted scoring
Logistic regression fusion
Stacked ensemble
Calibrated probability fusion
Attention-based fusion
```

---

# 16. Investigation Intelligence

A later project phase may introduce a Large Language Model.

The LLM will **not determine whether a customer is guilty or fraudulent**.

Instead:

```text
Verified Model Evidence
        │
Graph Evidence
        │
SHAP Explanation
        │
Sanctions Results
        │
Transaction Timeline
        │
        ▼
Investigation Assistant
        │
        ▼
Human-Readable Summary
```

Potential functions:

- investigation summaries;
- evidence organization;
- timeline descriptions;
- suspicious activity report assistance;
- natural-language exploration of cases.

Candidate technologies may later include:

```text
Local LLMs
HuggingFace Transformers
Ollama / LM Studio
RAG pipelines
Sentence Transformers
Vector databases
```

These are intentionally **not baseline dependencies**.

---

# 17. Directory Structure

```text
fincrime-intelligence/
│
├── backend/
│   │
│   ├── api/
│   │   ├── routes/
│   │   └── main.py
│   │
│   ├── core/
│   │   ├── config.py
│   │   ├── logging.py
│   │   └── security.py
│   │
│   ├── services/
│   │   ├── ingestion/
│   │   ├── features/
│   │   ├── graph/
│   │   ├── scoring/
│   │   ├── risk/
│   │   ├── explanations/
│   │   └── investigation/
│   │
│   └── tests/
│
├── research/
│   │
│   ├── notebooks/
│   │   ├── 00_dataset_exploration/
│   │   ├── 01_feature_analysis/
│   │   ├── 02_baselines/
│   │   ├── 03_graph/
│   │   ├── 04_gnn/
│   │   └── 05_temporal/
│   │
│   ├── experiments/
│   │   ├── baselines/
│   │   ├── graph_features/
│   │   ├── gnn/
│   │   └── temporal/
│   │
│   ├── models/
│   │   ├── tabular/
│   │   ├── anomaly/
│   │   ├── graph/
│   │   └── temporal/
│   │
│   ├── configs/
│   └── outputs/
│
├── frontend/
│   └── README.md
│
├── shared/
│   └── schemas/
│       ├── transaction.py
│       ├── entity.py
│       ├── risk.py
│       ├── alert.py
│       └── case.py
│
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── external/
│
├── scripts/
│
├── infrastructure/
│   ├── docker/
│   └── database/
│
├── requirements/
│   ├── base.txt
│   ├── research.txt
│   ├── graph.txt
│   ├── dev.txt
│   ├── optional.txt
│   └── locked.txt
│
├── docs/
│
├── .env.example
├── .gitignore
├── pyproject.toml
└── README.md
```

---

# 18. Directory Responsibilities

## `backend/`

Contains stable functionality intended to become part of the application.

Experimental code should not be written directly here.

---

## `research/`

Research sandbox.

Used for:

- notebooks;
- model experiments;
- feature experiments;
- GNN development;
- statistical analysis;
- temporal models;
- ablation studies.

---

## `frontend/`

Analyst-facing application.

Frontend development is intentionally independent of ML experimentation.

---

## `shared/`

Contains the contracts used throughout the platform.

Important entities include:

```text
Transaction
Entity
Prediction
Risk
Alert
Case
```

---

## `data/raw/`

Original datasets.

Raw data is immutable.

Never modify files inside this directory.

---

## `data/interim/`

Intermediate transformations.

---

## `data/processed/`

Validated, normalized, model-ready data.

---

## `data/external/`

External intelligence such as:

```text
Sanctions lists
Country risk data
Entity intelligence
Watchlists
```

---

## `requirements/`

Dependency groups.

Different research capabilities can therefore be installed without making every experimental technology mandatory.

---

# 19. Canonical Data Model

Different datasets have different structures.

The system therefore uses dataset adapters.

```text
IBM AML
    │
AMLSim
    │
Elliptic
    │
Future Dataset
    │
    ▼
DATASET ADAPTER
    │
    ▼
CANONICAL TRANSACTION SCHEMA
```

Example transaction representation:

```text
Transaction
│
├── transaction_id
├── timestamp
├── sender_id
├── receiver_id
├── amount
├── currency
├── payment_type
├── sender_bank
├── receiver_bank
├── sender_country
├── receiver_country
└── label
```

The remaining system therefore does not need to understand each original dataset format.

---

# 20. Entity Model

The system is designed to eventually represent multiple financial entity types.

```text
Entity
│
├── entity_id
├── entity_type
├── jurisdiction
├── account_age
├── organisation_type
├── risk_flags
└── external_identifiers
```

Future graph entities may include:

```text
PERSON
ACCOUNT
COMPANY
BANK
CRYPTO WALLET
JURISDICTION
TRANSACTION
```

This allows the same architecture to support both AML and sanctions research.

---

# 21. Data Pipeline

Initial pipeline:

```text
Raw Dataset
     │
     ▼
Dataset Adapter
     │
     ▼
Validation
     │
     ▼
Normalization
     │
     ▼
Canonical Transactions
     │
     ▼
Parquet
     │
     ├───────────────┐
     ▼               ▼
Feature Engine   Graph Builder
     │               │
     └───────┬───────┘
             ▼
           Models
             │
             ▼
         Evaluation
```

Later:

```text
Validated Model
       │
       ▼
Backend Service
       │
       ▼
FastAPI
       │
       ▼
Analyst Frontend
```

---

# 22. Feature Engineering

The first feature groups will include:

## Transaction Features

```text
Amount
Currency
Time
Payment Type
Cross-Border Indicator
```

---

## Behavioural Features

```text
Transaction Count
Average Transaction Amount
Historical Deviation
Unique Counterparties
Incoming / Outgoing Ratio
```

---

## Velocity Features

```text
Transactions per hour
Transactions per day
Time since previous transaction
Amount moved within short intervals
```

---

## Flow Features

```text
Incoming Amount
Outgoing Amount
Fund Retention Ratio
Fund Propagation Delay
```

---

## Graph Features

```text
In-degree
Out-degree
PageRank
Betweenness
Community
Cycle Participation
Fan-In
Fan-Out
Neighbour Risk
```

Research will determine which features provide meaningful predictive improvement.

---

# 23. Experimental Strategy

Experiments should follow a controlled progression.

### E001

```text
Logistic Regression
Transaction Features
```

### E002

```text
Random Forest
Transaction Features
```

### E003

```text
XGBoost
Transaction Features
```

### E004

```text
XGBoost
+
Behavioural Features
```

### E005

```text
XGBoost
+
Behavioural Features
+
Graph Features
```

### E006

```text
GCN
```

### E007

```text
GraphSAGE
```

Further experiments will be introduced only after evidence from the baseline results.

---

# 24. Evaluation Metrics

Financial crime datasets are typically heavily imbalanced.

Accuracy alone is therefore insufficient.

Primary metrics:

```text
Precision
Recall
F1 Score
PR-AUC
ROC-AUC
False Positive Rate
False Negative Rate
```

Operational metrics may eventually include:

```text
Alerts generated
Cases requiring review
False positives per 1,000 transactions
Inference time
Investigation time
Explanation stability
```

---

# 25. Dataset Splitting

Random splitting can introduce temporal leakage.

The preferred baseline is chronological splitting:

```text
Oldest 70%
    │
    ▼
TRAIN

Next 15%
    │
    ▼
VALIDATION

Newest 15%
    │
    ▼
TEST
```

Feature calculations must also avoid incorporating information from future transactions.

---

# 26. Experiment Reproducibility

Every experiment should receive an identifier.

Example:

```text
E005_xgb_graph_features
```

Each experiment records:

```text
Experiment ID
Dataset
Git Commit
Random Seed
Features
Model
Parameters
Data Split
Metrics
Runtime
Timestamp
```

Output example:

```text
research/outputs/E005_xgb_graph_features/
│
├── config.yaml
├── metrics.json
├── feature_importance.csv
├── confusion_matrix.png
└── notes.md
```

---

# 27. Configuration

Scientific decisions belong in YAML configuration.

Example:

```yaml
project:
  seed: 42

dataset:
  name: ibm_aml
  raw_path: data/raw/ibm_aml
  processed_path: data/processed/ibm_aml

split:
  strategy: chronological
  train_ratio: 0.70
  validation_ratio: 0.15
  test_ratio: 0.15

features:
  transaction: true
  behavioural: true
  graph: false
  temporal: false

model:
  type: xgboost

evaluation:
  metrics:
    - precision
    - recall
    - f1
    - roc_auc
    - pr_auc
```

Secrets and environment-specific configuration belong in `.env`.

---

# 28. Environment Variables

Example `.env.example`:

```env
APP_ENV=development

LOG_LEVEL=INFO

DATABASE_URL=postgresql+psycopg://fincrime:fincrime@localhost:5432/fincrime

ENTITY_HASH_SECRET=CHANGE_ME

NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=CHANGE_ME
```

Never commit the actual `.env` file.

---

# 29. Setup

## Clone the Repository

```bash
git clone <repository-url>
cd fincrime-intelligence
```

---

## Create the Python Environment

Windows:

```powershell
py -3.13 -m venv .venv
```

Activate:

```powershell
.venv\Scripts\Activate.ps1
```

Verify:

```powershell
python --version
```

Expected:

```text
Python 3.13.x
```

---

# 30. Install Dependencies

Upgrade packaging tools:

```powershell
python -m pip install --upgrade pip setuptools wheel
```

Install baseline:

```powershell
pip install -r requirements\base.txt
```

Install research tools:

```powershell
pip install -r requirements\research.txt
```

Install graph tools:

```powershell
pip install -r requirements\graph.txt
```

Install development tools:

```powershell
pip install -r requirements\dev.txt
```

Verify:

```powershell
pip check
```

Expected:

```text
No broken requirements found.
```

---

# 31. Jupyter Environment

Register the virtual environment:

```powershell
python -m ipykernel install --user --name fincrime --display-name "FinCrime Research"
```

Launch:

```powershell
jupyter lab
```

Select:

```text
FinCrime Research
```

as the notebook kernel.

---

# 32. PyTorch and Graph Neural Networks

PyTorch is intentionally installed separately because installation may depend on CPU/GPU configuration.

Verify after installation:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Then install PyTorch Geometric when graph neural network experimentation begins.

Graph ML is not required to run the baseline platform.

---

# 33. Development Workflow

Recommended branches:

```text
main
│
└── develop
     │
     ├── feature/data-pipeline
     ├── feature/backend-api
     ├── feature/frontend
     ├── feature/security
     ├── research/graph-features
     └── research/gnn
```

Avoid committing unfinished experiments directly to `main`.

---

# 34. Team

The project is being developed by:

### Johan

Primary areas:

```text
System Architecture
Graph Intelligence
Risk Fusion
ML / AI
Research Coordination
```

---

### Atharva

Primary areas:

```text
Data Engineering
Feature Engineering
Machine Learning
Experimentation
Model Evaluation
```

---

### Liston

Primary areas:

```text
Frontend Development
Analyst Experience
Dashboard Design
Graph Visualization
Explainability Presentation
```

---

### Ansh

Primary areas:

```text
Backend APIs
Cybersecurity
Privacy Architecture
Infrastructure
Deployment
Authentication / Authorization
```

All members are expected to understand the overall architecture even where subsystem ownership differs.

---

# 35. Current Development Roadmap

## Phase 0 — Foundation

```text
Repository
Python Environment
Configuration
Directory Structure
Schemas
```

---

## Phase 1 — Data

```text
Dataset Selection
Dataset Adapter
Canonical Schema
Validation
Parquet Conversion
EDA
```

---

## Phase 2 — ML Baseline

```text
Feature Engineering
Logistic Regression
Random Forest
XGBoost
Evaluation
```

---

## Phase 3 — Graph Intelligence

```text
Transaction Graph
NetworkX
Graph Features
Community Detection
Cycle Analysis
Graph-Augmented ML
```

---

## Phase 4 — Advanced Graph Learning

```text
PyTorch Geometric
GCN
GraphSAGE
GAT
```

---

## Phase 5 — Application Platform

```text
FastAPI
Risk Engine
Alert Engine
Case Management
PostgreSQL
Neo4j
```

---

## Phase 6 — Analyst Interface

```text
React Dashboard
Alert Queue
Investigation Workspace
Interactive Graph
Risk Explanations
```

---

## Phase 7 — Research Specialization

Based on experimental findings:

```text
AML
        OR

Sanctions Evasion
        OR

General Financial Crime Intelligence
```

---

# 36. Potential Research Directions

The platform is deliberately designed to investigate several open questions.

### Graph Intelligence

How much does transaction-network structure improve prediction compared with tabular ML?

---

### Temporal Graphs

Can the timing of fund propagation reveal laundering structures missed by static graphs?

---

### Risk Fusion

Can combining behavioural, tabular, graph, anomaly, and external intelligence outperform individual models?

---

### Explainability

Are explanations from financial crime models stable and faithful?

---

### False Positive Reduction

Can graph and behavioural context reduce unnecessary AML alerts?

---

### Adaptive Detection

How should fraud detection systems respond when criminal behaviour changes?

---

### Sanctions Networks

Can knowledge graphs detect indirect relationships with sanctioned entities?

---

# 37. What This Project Is Not

This project is **not** intended to:

- determine criminal guilt;
- autonomously sanction individuals;
- replace financial investigators;
- blindly trust LLM-generated analysis;
- treat model probability as definitive evidence.

The platform is designed as:

> A Financial Crime Risk Intelligence and Investigation Support System.

Human investigation remains part of the decision process.

---

# 38. Baseline Success Criteria

The first complete baseline should support:

```text
Raw Dataset
     ↓
Canonical Transactions
     ↓
Feature Engineering
     ↓
ML Baseline
     ↓
Transaction Graph
     ↓
Graph Features
     ↓
Risk Prediction
     ↓
Explanation
```

The application baseline then extends this into:

```text
Prediction
    ↓
Risk Score
    ↓
Alert
    ↓
Case
    ↓
Analyst Dashboard
```

---

# 39. Long-Term Vision

The final platform should eventually allow researchers and analysts to ask:

```text
Which transactions are suspicious?

Why were they considered suspicious?

Which entities are connected?

How did funds move across the network?

Does the behaviour resemble known financial crime patterns?

Is the entity indirectly connected to high-risk entities?

Which evidence contributed to the risk score?

How reliable is the model's explanation?

What should an investigator inspect next?
```

The system should not simply produce predictions.

It should transform raw financial activity into **structured, explainable, and investigable risk intelligence**.

---

# FinCrime Intelligence

```text
Transactions
     +
Behaviour
     +
Networks
     +
Machine Learning
     +
Explainability
     +
Human Investigation

              ↓

      Financial Crime Intelligence
```

**Research first. Modular by design. Evidence before complexity.**