"""
PyTorch Geometric GNN Models for Fraud Ring Node & Edge Classification:
  1. FraudGraphSAGE: Inductive neighborhood aggregation (SAGEConv)
  2. FraudGAT: Multi-head Graph Attention Network (GATConv) with edge attributes
Includes Weighted Cross-Entropy and Focal Loss for severe class imbalance (0.5%-2% fraud).
"""

from __future__ import annotations

from typing import Literal, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv


class FocalLoss(nn.Module):
    """
    Binary Focal Loss for extreme fraud class imbalance:
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

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
        focal_weight = alpha_t * torch.pow(1.0 - pt, self.gamma)
        return torch.mean(focal_weight * bce)


class WeightedBCELoss(nn.Module):
    """Weighted binary cross-entropy loss for rare positive fraud classes."""

    def __init__(self, pos_weight: float = 25.0):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pw = torch.tensor([self.pos_weight], device=logits.device, dtype=logits.dtype)
        return F.binary_cross_entropy_with_logits(logits, targets.float(), pos_weight=pw)


class FraudGraphSAGE(nn.Module):
    """2-Layer inductive GraphSAGE model for node & edge fraud scoring."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr="mean")
        self.conv2 = SAGEConv(hidden_channels, hidden_channels, aggr="max")
        self.node_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 1),
        )
        self.edge_head = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1),
        )
        self.dropout = dropout

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index)
        return F.relu(h)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.encode(x, edge_index)
        return self.node_head(h).squeeze(-1)

    def predict_edges(
        self, x: torch.Tensor, edge_index: torch.Tensor, query_edges: torch.Tensor
    ) -> torch.Tensor:
        h = self.encode(x, edge_index)
        src_h = h[query_edges[0]]
        dst_h = h[query_edges[1]]
        return self.edge_head(torch.cat([src_h, dst_h], dim=-1)).squeeze(-1)


class FraudGAT(nn.Module):
    """Multi-head Graph Attention Network (GAT) with returned attention weights."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 32,
        heads: int = 4,
        dropout: float = 0.30,
    ):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.conv2 = GATConv(
            hidden_channels * heads,
            hidden_channels,
            heads=1,
            concat=False,
            dropout=dropout,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ELU(),
            nn.Linear(hidden_channels // 2, 1),
        )
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        return_attention: bool = False,
    ):
        h = F.dropout(x, p=self.dropout, training=self.training)
        h = F.elu(self.conv1(h, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        if return_attention:
            h, (attn_edge_index, attn_weights) = self.conv2(
                h, edge_index, return_attention_weights=True
            )
            logits = self.classifier(F.elu(h)).squeeze(-1)
            return logits, attn_edge_index, attn_weights
        h = F.elu(self.conv2(h, edge_index))
        return self.classifier(h).squeeze(-1)


def build_loss(
    loss_type: Literal["focal", "weighted_ce"] = "focal",
    pos_weight: float = 25.0,
    focal_alpha: float = 0.85,
    focal_gamma: float = 2.0,
) -> nn.Module:
    if loss_type == "focal":
        return FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    return WeightedBCELoss(pos_weight=pos_weight)
