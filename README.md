# fraud-ring-finder

**Graph Machine Learning Transaction Forensics Workbench & Synthetic Fraud Ring Benchmark**

![Fraud Ring Finder — Interactive Network Workbench](docs/images/ui_workbench_overview.png)

`fraud-ring-finder` is a desktop-grade analytical application and Python/PyTorch Geometric codebase that detects organized fraud rings in financial transaction networks. It pairs a **deterministic, seeded synthetic network generator** (planting four canonical fraud ring topologies) with **XGBoost baselines**, **GraphSAGE**, **GAT (Graph Attention Networks)**, **SHAP TreeExplainer**, and **GNNExplainer 2-hop subgraph attributions**.

---

## 1. Visual Interface Preview

### Interactive Network Topology (PyVis) & Per-Account Risk Card
Highlights planted **Money-Mule Chains**, **Shared-Device Rings**, **High-Velocity Burst Clusters**, and **Circular Transfers** alongside real-time **SHAP log-odds feature attributions** and **GNNExplainer 2-hop edge masks**.

![Network Workbench Overview](docs/images/ui_workbench_overview.png)

### Evaluation Protocol, Per-Pattern Ablation & Post-Training Distribution Shift
Compares **PR-AUC**, **Recall at Fixed 80% Precision**, per-pattern recall breakdown, and robustness when fraudsters rotate devices and throttle burst velocity after the training split (`tactic_shift`).

![Evaluation and Ablation Study](docs/images/ui_evaluation_ablation.png)

---

## 2. Why Synthetic Data & What That Implies

### Why Synthetic?
1. **Banking Secrecy & PII Constraints**: Real wire, ACH, and P2P transaction graphs containing device fingerprints (`device_id`), IP subnets, and confirmed multi-account fraud rings cannot be redistributed publicly under GLBA, GDPR, and PCI-DSS regulations.
2. **Ground-Truth Ring Topology**: Publicly available fraud datasets (e.g., anonymized credit card tables) either strip graph edges into PCA vectors or label individual chargebacks without identifying **which accounts belong to the same collusive ring**.
3. **Controlled Ablation & Distribution Shift**: A configurable generator allows us to isolate *exactly* which structural pattern (**Money-Mule Layering Chains** vs **Shared-Device Rings** vs **Burst Clusters** vs **Circular Wash Loops**) requires hand-crafted graph features vs multi-hop GNN message passing, and to test post-training tactic shifts without confounding noise.

### What That Implies
- **Cleaner Background Noise**: Synthetic legitimate transactions follow a log-normal amount distribution and Poisson arrival process with a 12% power-user cohort. Real production ledgers contain complex non-fraudulent structures (corporate payroll fan-out, marketplace escrow hubs, treasury sweeps) that increase false-positive pressure at high precision targets.
- **Constructive Topology**: Because the planted rings follow explicit parametric rules in `src/generator/synthetic_network.py`, high recall on a synthetic pattern proves that a model class has the **expressive capacity** to represent that topology—not that hyperparameters are pre-tuned for a specific bank's production distribution.

---

## 3. How to Plug In Real Transaction Data

To run `fraud-ring-finder` on your own payment ledger instead of the synthetic generator, supply two DataFrames (CSV or Parquet) matching the schema consumed by `src/features/graph_features.py`:

### Required Schema

1. **`accounts.parquet`**:
   - `account_id` (`str`): Unique account identifier
   - `account_age_days` (`int`): Account tenure at observation cutoff
   - `kyc_tier` (`int`): Verification level (`1`, `2`, `3`)
   - `declared_monthly_vol` (`float`): Expected monthly volume declared at onboarding

2. **`transactions.parquet`**:
   - `tx_id` (`str`): Unique transaction ID
   - `src_id` (`str`), `dst_id` (`str`): Sender and receiver `account_id`
   - `device_id` (`str`): Device fingerprint hash or session token
   - `merchant_id` (`str`): Counterparty merchant identifier
   - `timestamp_hr` (`float`): Monotonic timestamp in hours from epoch/start window
   - `amount` (`float`): Transaction value in base currency
   - `is_fraud` (`int`): Ground-truth binary label (`0` or `1`)

### Drop-in Integration Snippet

```python
import pandas as pd
from src.features.graph_features import (
    temporal_train_val_test_split,
    build_features_up_to_time,
)
from src.models.baselines import train_xgboost_baseline
from src.evaluation.metrics import evaluate_model_predictions

accounts_df = pd.read_parquet("data/accounts.parquet")
transactions_df = pd.read_parquet("data/transactions.parquet")

# 1. Strict temporal split (zero future data leakage)
train_cut, val_cut, train_tx, val_tx, test_tx = temporal_train_val_test_split(
    transactions_df, train_ratio=0.60, val_ratio=0.20
)

# 2. Extract tabular + graph topology features strictly up to each cutoff
train_feat = build_features_up_to_time(accounts_df, transactions_df, max_timestamp_hr=train_cut)
test_feat = build_features_up_to_time(accounts_df, transactions_df, max_timestamp_hr=transactions_df["timestamp_hr"].max())

# 3. Train XGBoost + Graph Features and evaluate PR-AUC & Recall @ 80% Precision
artifact = train_xgboost_baseline(train_feat, test_feat, include_graph_features=True)
summary = evaluate_model_predictions("XGBoost + Graph Features", test_feat, artifact.probs, target_precision=0.80)
print(summary)
```

---

## 4. Script-Generated Benchmark & Ablation Results

> **Reproducibility Note**: Every metric in the tables below was generated directly by running `npx tsx scripts/generate_readme.ts` with `seed=42`, `num_accounts=96`, `num_transactions=540`, `fraud_rate=0.015`, `train_cutoff=102.12h`, and `val_cutoff=134.953h`. No numbers are hand-invented.

### 4.1 Overall Performance & Distribution Shift Robustness

| Model Architecture | Imbalance Strategy | PR-AUC (Stationary) | Recall @ 80% Precision | Post-Shift PR-AUC (`tactic_shift`) | Shift Drop |
| :--- | :--- | ---: | ---: | ---: | ---: |
| **XGBoost (Tabular Baseline)** | Weighted Log-Loss (scale_pos_weight) | `0.4939` | `25.0%` | `0.4939` | `-0.0%` |
| **XGBoost + Graph Features** | Weighted Log-Loss (scale_pos_weight) | `0.7203` | `50.0%` | `0.6115` | `-15.1%` |
| **GraphSAGE (2-Layer Inductive)** | Focal Loss (α=0.85, γ=2.0) | `0.7998` | `75.0%` | `0.5153` | `-35.6%` |
| **GAT (4-Head Graph Attention)** | Weighted Cross-Entropy (pos_weight=25) | `0.7754` | `75.0%` | `0.5797` | `-25.7%` |

### 4.2 Per-Pattern Detection Ablation (Recall at 80% Precision Threshold)

| Planted Fraud Ring Pattern | Ring ID | XGBoost (Tabular) | XGBoost + Graph Features | GraphSAGE (Focal Loss) | GAT (Weighted CE) |
| :--- | :--- | ---: | ---: | ---: | ---: |
| **High-Velocity Burst Cluster** | `RING-BURST-03` | `100%` | `100%` | `100%` | `100%` |
| **Shared-Device Collusion Ring** | `RING-SDEV-02` | `0%` | `100%` | `100%` | `100%` |
| **Circular Wash Transfer Loop** | `RING-CYCLE-04` | `0%` | `0%` | `100%` | `100%` |
| **Money-Mule 5-Hop Layering Chain** | `RING-MULE-01` | `0%` | `0%` | `0%` | `0%` |

---

## 5. Project Structure

```text
.
├── configs/
│   └── default.yaml                      # Seeded generator, split, XGBoost, GraphSAGE & GAT config
├── docs/
│   └── images/
│       ├── ui_workbench_overview.png     # Captured screenshot of the Network Workbench UI
│       └── ui_evaluation_ablation.png    # Captured screenshot of the Evaluation & Ablation UI
├── scripts/
│   └── generate_readme.ts                # Script that computes all benchmark numbers & writes README.md
├── src/
│   ├── generator/
│   │   └── synthetic_network.py          # Configurable seeded NetworkX fraud ring generator
│   ├── features/
│   │   └── graph_features.py             # Temporal split & zero-leakage tabular + graph feature builder
│   ├── models/
│   │   ├── baselines.py                  # XGBoost Tabular & Graph baselines + SHAP TreeExplainer
│   │   └── gnn_models.py                 # PyTorch Geometric GraphSAGE, GAT, FocalLoss & WeightedBCELoss
│   ├── evaluation/
│   │   └── metrics.py                    # PR-AUC, Recall @ fixed precision, pattern ablation
│   ├── explainability/
│   │   └── explainer.py                  # GNNExplainer 2-hop subgraph & node attribute masks
│   ├── app_streamlit.py                  # Streamlit + PyVis interactive fraud ring dashboard
│   ├── engine/
│   │   ├── fraudEngine.ts                # Deterministic TypeScript simulation & GNN/SHAP engine
│   │   └── fraudEngine.test.ts           # Automated determinism & zero-leakage test suite
│   ├── components/                       # Desktop Workbench React UI components
│   └── App.tsx                           # Main Desktop Application shell
├── tests/
│   └── test_generator_and_leakage.py     # Python unit tests for generator determinism & no-leakage
└── README.md                             # Script-generated documentation
```

---

## 6. Running Tests & Application

```bash
# 1. Run deterministic generator & zero-leakage unit tests
npx --yes tsx --test src/engine/fraudEngine.test.ts
python3 -m unittest tests/test_generator_and_leakage.py

# 2. Regenerate README benchmark tables from live script execution
npx --yes tsx scripts/generate_readme.ts

# 3. Launch Desktop Web Workbench (Port 3000) or Streamlit App
npm run dev
streamlit run src/app_streamlit.py
```
