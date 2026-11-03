"""
Explainability Module:
  - SHAP feature importance for XGBoost models
  - GNNExplainer subgraph & feature mask extraction for PyTorch Geometric GNNs
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import torch
from torch_geometric.explain import Explainer, GNNExplainer


@dataclass
class SubgraphExplanation:
    target_node_idx: int
    top_edge_pairs: List[Tuple[int, int, float]]
    top_feature_weights: Dict[str, float]


def explain_node_with_gnn_explainer(
    model: torch.nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    target_node_idx: int,
    feature_names: List[str],
    top_k_edges: int = 8,
    epochs: int = 80,
) -> SubgraphExplanation:
    """
    Runs PyTorch Geometric's GNNExplainer to identify the minimal 2-hop subgraph
    and node features driving the fraud prediction for `target_node_idx`.
    """
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=epochs),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=dict(
            mode="binary_classification",
            task_level="node",
            return_type="raw",
        ),
    )

    explanation = explainer(x, edge_index, index=target_node_idx)
    edge_mask = explanation.edge_mask.detach().cpu().numpy()
    node_mask = explanation.node_mask.detach().cpu().numpy()

    top_edge_indices = edge_mask.argsort()[::-1][:top_k_edges]
    edge_pairs: List[Tuple[int, int, float]] = []
    for idx in top_edge_indices:
        score = float(edge_mask[idx])
        if score > 0.01:
            u = int(edge_index[0, idx].item())
            v = int(edge_index[1, idx].item())
            edge_pairs.append((u, v, round(score, 4)))

    target_feat_scores = node_mask[target_node_idx]
    top_feat_indices = target_feat_scores.argsort()[::-1][:6]
    feat_weights = {
        feature_names[i]: round(float(target_feat_scores[i]), 4)
        for i in top_feat_indices
    }

    return SubgraphExplanation(
        target_node_idx=target_node_idx,
        top_edge_pairs=edge_pairs,
        top_feature_weights=feat_weights,
    )
