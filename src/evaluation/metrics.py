"""
Evaluation Protocol:
  - PR-AUC (Precision-Recall Area Under Curve)
  - Recall at Fixed Precision (e.g., Recall @ 80% Precision)
  - Per-Pattern Detection Ablation (money_mule_chain, shared_device_ring,
    burst_activity, circular_transfer)
  - Distribution Shift Robustness (Pre-shift vs Tactic Shift)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve


@dataclass
class EvaluationSummary:
    model_name: str
    pr_auc: float
    recall_at_fixed_precision: float
    fixed_precision_target: float
    pattern_recall: Dict[str, float]


def recall_at_precision(
    y_true: np.ndarray,
    y_score: np.ndarray,
    target_precision: float = 0.80,
) -> float:
    precisions, recalls, _ = precision_recall_curve(y_true, y_score)
    valid_mask = precisions >= target_precision
    if not np.any(valid_mask):
        return 0.0
    return float(np.max(recalls[valid_mask]))


def evaluate_model_predictions(
    model_name: str,
    eval_df: pd.DataFrame,
    y_score: np.ndarray,
    target_precision: float = 0.80,
) -> EvaluationSummary:
    y_true = eval_df["is_fraud"].to_numpy(dtype=np.int32)
    pr_auc = float(average_precision_score(y_true, y_score))
    rec_fixed = recall_at_precision(y_true, y_score, target_precision=target_precision)

    # Determine threshold achieving target_precision
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
    valid_indices = np.where(precisions[:-1] >= target_precision)[0]
    if len(valid_indices) > 0:
        best_idx = valid_indices[np.argmax(recalls[valid_indices])]
        threshold = float(thresholds[best_idx])
    else:
        threshold = 0.50

    preds = (y_score >= threshold).astype(np.int32)
    patterns = [
        "money_mule_chain",
        "shared_device_ring",
        "burst_activity",
        "circular_transfer",
    ]
    pattern_recall: Dict[str, float] = {}
    for pat in patterns:
        mask = (eval_df["fraud_pattern"] == pat).to_numpy()
        if np.sum(mask) == 0:
            pattern_recall[pat] = 0.0
        else:
            pattern_recall[pat] = round(float(np.mean(preds[mask] == 1)), 4)

    return EvaluationSummary(
        model_name=model_name,
        pr_auc=round(pr_auc, 4),
        recall_at_fixed_precision=round(rec_fixed, 4),
        fixed_precision_target=target_precision,
        pattern_recall=pattern_recall,
    )
