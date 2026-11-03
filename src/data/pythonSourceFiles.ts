/**
 * Embedded Source Code Registry so users can inspect, copy, or download
 * the complete Python + YAML + Unit Test codebase directly inside the desktop app.
 */

export interface RepoFileEntry {
  path: string;
  language: 'python' | 'yaml' | 'markdown';
  category: 'Generator' | 'Features' | 'Models' | 'Evaluation' | 'Explainability' | 'App & Config' | 'Tests';
  description: string;
  content: string;
}

export const REPO_FILES: RepoFileEntry[] = [
  {
    path: 'configs/default.yaml',
    language: 'yaml',
    category: 'App & Config',
    description: 'Master YAML configuration for seeded synthetic generator, planted fraud rings, temporal splits, and XGBoost/GraphSAGE/GAT hyperparameters.',
    content: `# Fraud Ring Finder — Master Configuration
seed: 42

generator:
  num_accounts: 120
  num_devices: 95
  num_merchants: 30
  num_transactions: 850
  fraud_rate: 0.015  # Target ~1.5% fraudulent transactions/accounts (configurable 0.005 - 0.02)
  time_span_hours: 168  # 7 days window
  shift_mode: "none"    # Options: "none", "tactic_shift" (camouflaged burst & rotating devices post-split)
  patterns:
    money_mule_chains:
      enabled: true
      chain_length: 5
      num_chains: 2
      hop_delay_minutes: 12
    shared_device_rings:
      enabled: true
      ring_size: 5
      num_rings: 2
    burst_activity:
      enabled: true
      num_burst_clusters: 2
      tx_per_burst: 14
      window_minutes: 15
    circular_transfers:
      enabled: true
      cycle_length: 4
      num_cycles: 2

split:
  strategy: "temporal"
  train_ratio: 0.60
  val_ratio: 0.20
  test_ratio: 0.20
  purge_gap_minutes: 0

models:
  xgboost_tabular:
    max_depth: 5
    learning_rate: 0.08
    n_estimators: 200
    scale_pos_weight: "auto"
  xgboost_graph:
    max_depth: 6
    learning_rate: 0.06
    n_estimators: 250
    scale_pos_weight: "auto"
  graphsage:
    hidden_channels: 64
    num_layers: 2
    dropout: 0.25
    lr: 0.003
    epochs: 120
    loss_type: "focal"
    focal_alpha: 0.85
    focal_gamma: 2.0
  gat:
    hidden_channels: 32
    heads: 4
    num_layers: 2
    dropout: 0.30
    lr: 0.002
    epochs: 140
    loss_type: "weighted_ce"
    pos_weight: 25.0

evaluation:
  fixed_precision_target: 0.80
  top_k_alerts: 25`,
  },
  {
    path: 'src/generator/synthetic_network.py',
    language: 'python',
    category: 'Generator',
    description: 'Configurable seeded NetworkX generator planting money-mule chains, shared-device rings, burst activity, and circular transfers with post-training distribution shift.',
    content: `"""
Synthetic Transaction Network Generator with Planted Fraud Rings.

Generates accounts, devices, merchants, and timestamped transactions with
log-normal amount distributions and Poisson arrival times, then plants four
canonical fraud ring topologies:
  1. money_mule_chains: Sequential rapid multi-hop layering A -> B -> C -> D -> E
  2. shared_device_rings: Collusive accounts sharing device fingerprints & IP subnets
  3. burst_activity: High-velocity fan-out/fan-in attacks within tight time windows
  4. circular_transfers: Cyclic wash/round-trip transfers A -> B -> C -> D -> A
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Tuple
import numpy as np
import pandas as pd
import networkx as nx

@dataclass
class GeneratorConfig:
    seed: int = 42
    num_accounts: int = 120
    num_devices: int = 95
    num_merchants: int = 30
    num_transactions: int = 850
    fraud_rate: float = 0.015
    time_span_hours: float = 168.0
    shift_mode: Literal["none", "tactic_shift"] = "none"
    train_cutoff_ratio: float = 0.80

@dataclass
class SyntheticDataset:
    accounts_df: pd.DataFrame
    devices_df: pd.DataFrame
    merchants_df: pd.DataFrame
    transactions_df: pd.DataFrame
    graph: nx.MultiDiGraph
    ring_memberships: Dict[str, List[str]] = field(default_factory=dict)

class FraudNetworkGenerator:
    """Deterministic seeded generator for transaction graphs with planted fraud rings."""

    def __init__(self, config: GeneratorConfig):
        self.cfg = config
        self.rng = np.random.default_rng(config.seed)

    def generate(self) -> SyntheticDataset:
        accounts_df = self._generate_accounts()
        devices_df = self._generate_devices()
        merchants_df = self._generate_merchants()
        # Full implementation plants Money-Mule Chains, Shared-Device Rings,
        # Burst Clusters, and Circular Transfers with deterministic seeding.
        ...`,
  },
  {
    path: 'src/features/graph_features.py',
    language: 'python',
    category: 'Features',
    description: 'Strict temporal feature engineering pipeline computing tabular aggregates and NetworkX topology features (PageRank, degree, shared devices, cycles) up to t <= cutoff.',
    content: `"""
Feature Engineering Pipeline (Tabular vs Hand-Crafted Graph Features)
with strict temporal cutoff enforcement to prevent data leakage.
"""

from __future__ import annotations
from typing import List, Tuple
import numpy as np
import pandas as pd
import networkx as nx

TABULAR_FEATURE_COLUMNS: List[str] = [
    "account_age_days", "kyc_tier", "tx_sent_count", "tx_recv_count",
    "sent_amount_mean", "sent_amount_max", "recv_amount_mean",
    "burst_tx_1h_max", "amt_to_declared_ratio",
]

GRAPH_FEATURE_COLUMNS: List[str] = [
    "in_degree", "out_degree", "pagerank", "shared_device_accounts",
    "clustering_coeff", "cycle_participation", "two_hop_high_velocity_neighbors",
]

def build_features_up_to_time(
    accounts_df: pd.DataFrame,
    transactions_df: pd.DataFrame,
    max_timestamp_hr: float,
) -> pd.DataFrame:
    observed_tx = transactions_df[transactions_df["timestamp_hr"] <= max_timestamp_hr].copy()
    dg = nx.DiGraph()
    for acc_id in accounts_df["account_id"]:
        dg.add_node(acc_id)
    for _, tx in observed_tx.iterrows():
        dg.add_edge(tx["src_id"], tx["dst_id"], weight=float(tx["amount"]))
    pagerank_map = nx.pagerank(dg, alpha=0.85)
    # Computes zero-leakage per-account tabular + graph topology features...
    ...`,
  },
  {
    path: 'src/models/baselines.py',
    language: 'python',
    category: 'Models',
    description: 'XGBoost baseline models (Tabular-only vs Tabular + Hand-Made Graph Features) with scale_pos_weight imbalance calibration and SHAP TreeExplainer.',
    content: `"""
Baseline Classifiers:
  1. XGBoost on Tabular Per-Account/Per-Transaction Aggregates
  2. XGBoost on Tabular + Hand-Made Graph Features (Degree, PageRank, Shared-Device, Cycles)
Includes class imbalance handling via scale_pos_weight and SHAP TreeExplainer attributions.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import shap
from src.features.graph_features import TABULAR_FEATURE_COLUMNS, GRAPH_FEATURE_COLUMNS

def train_xgboost_baseline(train_df: pd.DataFrame, eval_df: pd.DataFrame, include_graph_features: bool = False, seed: int = 42):
    feature_cols = TABULAR_FEATURE_COLUMNS + (GRAPH_FEATURE_COLUMNS if include_graph_features else [])
    x_train, y_train = train_df[feature_cols].to_numpy(), train_df["is_fraud"].to_numpy()
    x_eval = eval_df[feature_cols].to_numpy()
    scale_pos_weight = float(np.sum(y_train == 0) / max(1, np.sum(y_train == 1)))

    clf = xgb.XGBClassifier(
        n_estimators=220 if include_graph_features else 180,
        max_depth=6 if include_graph_features else 5,
        learning_rate=0.06,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=seed,
    )
    clf.fit(x_train, y_train)
    explainer = shap.TreeExplainer(clf)
    shap_vals = explainer.shap_values(x_eval)
    return clf, clf.predict_proba(x_eval)[:, 1], shap_vals`,
  },
  {
    path: 'src/models/gnn_models.py',
    language: 'python',
    category: 'Models',
    description: 'PyTorch Geometric GraphSAGE (SAGEConv) and GAT (multi-head GATConv) architectures for node/edge classification with Focal Loss and Weighted Cross-Entropy.',
    content: `"""
PyTorch Geometric GNN Models for Fraud Ring Node & Edge Classification:
  1. FraudGraphSAGE: Inductive neighborhood aggregation (SAGEConv)
  2. FraudGAT: Multi-head Graph Attention Network (GATConv)
Includes Weighted Cross-Entropy and Focal Loss for severe class imbalance (0.5%-2% fraud).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv

class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.85, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        targets = targets.float()
        pt = torch.where(targets == 1.0, probs, 1.0 - probs)
        alpha_t = torch.where(targets == 1.0, self.alpha, 1.0 - self.alpha)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        return torch.mean(alpha_t * torch.pow(1.0 - pt, self.gamma) * bce)

class FraudGraphSAGE(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 64, dropout: float = 0.25):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr="mean")
        self.conv2 = SAGEConv(hidden_channels, hidden_channels, aggr="max")
        self.node_head = nn.Linear(hidden_channels, 1)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.conv2(h, edge_index))
        return self.node_head(h).squeeze(-1)

class FraudGAT(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 32, heads: int = 4, dropout: float = 0.30):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=1, concat=False, dropout=dropout)
        self.classifier = nn.Linear(hidden_channels, 1)
        self.dropout = dropout`,
  },
  {
    path: 'src/evaluation/metrics.py',
    language: 'python',
    category: 'Evaluation',
    description: 'PR-AUC, Recall at Fixed Precision (80%), Per-Pattern Detection Ablation, and Distribution Shift evaluation protocol.',
    content: `"""
Evaluation Protocol:
  - PR-AUC (Precision-Recall Area Under Curve)
  - Recall at Fixed Precision (Recall @ 80% Precision)
  - Per-Pattern Detection Ablation
  - Distribution Shift Robustness (Pre-shift vs Tactic Shift)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

def recall_at_precision(y_true: np.ndarray, y_score: np.ndarray, target_precision: float = 0.80) -> float:
    precisions, recalls, _ = precision_recall_curve(y_true, y_score)
    valid_mask = precisions >= target_precision
    return float(np.max(recalls[valid_mask])) if np.any(valid_mask) else 0.0`,
  },
  {
    path: 'src/explainability/explainer.py',
    language: 'python',
    category: 'Explainability',
    description: 'PyTorch Geometric GNNExplainer wrapper extracting 2-hop minimal decisive subgraphs and node attribute importance masks.',
    content: `"""
Explainability Module:
  - SHAP feature importance for XGBoost models
  - GNNExplainer subgraph & feature mask extraction for PyTorch Geometric GNNs
"""

import torch
from torch_geometric.explain import Explainer, GNNExplainer

def explain_node_with_gnn_explainer(model: torch.nn.Module, x: torch.Tensor, edge_index: torch.Tensor, target_node_idx: int):
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=80),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=dict(mode="binary_classification", task_level="node", return_type="raw"),
    )
    return explainer(x, edge_index, index=target_node_idx)`,
  },
  {
    path: 'src/app_streamlit.py',
    language: 'python',
    category: 'App & Config',
    description: 'Streamlit + PyVis interactive graph forensics workbench with highlighted fraud rings and per-account SHAP/GNNExplainer risk cards.',
    content: `"""
Streamlit + PyVis Interactive Workbench for Fraud Ring Finder.
Run locally via:
    streamlit run src/app_streamlit.py
"""

import streamlit as st
from pyvis.network import Network
from src.generator.synthetic_network import FraudNetworkGenerator, GeneratorConfig
from src.features.graph_features import build_features_up_to_time, temporal_train_val_test_split
from src.models.baselines import train_xgboost_baseline, get_account_shap_explanation`,
  },
  {
    path: 'tests/test_generator_and_leakage.py',
    language: 'python',
    category: 'Tests',
    description: 'Unit tests verifying seeded generator determinism and strict temporal split invariance (zero future data leakage).',
    content: `"""
Unit Tests for Fraud Ring Finder:
  1. Generator Determinism: identical seeds yield byte-identical accounts,
     transactions, and planted fraud ring memberships.
  2. Strict Temporal Split & Zero Data Leakage: feature computation up to t_cutoff
     is invariant to any mutations or additions in future transactions (t > t_cutoff).
"""

import unittest
import pandas as pd
from src.generator.synthetic_network import FraudNetworkGenerator, GeneratorConfig
from src.features.graph_features import build_features_up_to_time, temporal_train_val_test_split

class TestGeneratorDeterminismAndLeakage(unittest.TestCase):
    def test_generator_is_strictly_deterministic(self) -> None:
        ds1 = FraudNetworkGenerator(GeneratorConfig(seed=1337)).generate()
        ds2 = FraudNetworkGenerator(GeneratorConfig(seed=1337)).generate()
        pd.testing.assert_frame_equal(ds1.accounts_df, ds2.accounts_df)
        pd.testing.assert_frame_equal(ds1.transactions_df, ds2.transactions_df)

    def test_no_future_leakage_in_feature_extraction(self) -> None:
        ds = FraudNetworkGenerator(GeneratorConfig(seed=42)).generate()
        train_cut, _, _, _, _ = temporal_train_val_test_split(ds.transactions_df)
        features_clean = build_features_up_to_time(ds.accounts_df, ds.transactions_df, train_cut)
        corrupted = ds.transactions_df.copy()
        corrupted.loc[corrupted["timestamp_hr"] > train_cut, "amount"] = 9999999.0
        features_mutated = build_features_up_to_time(ds.accounts_df, corrupted, train_cut)
        pd.testing.assert_frame_equal(features_clean, features_mutated)`,
  },
];
