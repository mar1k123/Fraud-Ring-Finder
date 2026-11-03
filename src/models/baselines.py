"""
Baseline Classifiers:
  1. XGBoost on Tabular Per-Account/Per-Transaction Aggregates
  2. XGBoost on Tabular + Hand-Made Graph Features (Degree, PageRank, Shared-Device, Cycles)
Includes class imbalance handling via `scale_pos_weight` and SHAP TreeExplainer attributions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import numpy as np
import pandas as pd
import xgboost as xgb
import shap

from src.features.graph_features import TABULAR_FEATURE_COLUMNS, GRAPH_FEATURE_COLUMNS


@dataclass
class XGBModelArtifact:
    name: str
    model: xgb.XGBClassifier
    feature_names: List[str]
    probs: np.ndarray
    shap_values: np.ndarray


def train_xgboost_baseline(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    include_graph_features: bool = False,
    seed: int = 42,
) -> XGBModelArtifact:
    feature_cols = list(TABULAR_FEATURE_COLUMNS)
    if include_graph_features:
        feature_cols = feature_cols + list(GRAPH_FEATURE_COLUMNS)

    x_train = train_df[feature_cols].to_numpy(dtype=np.float32)
    y_train = train_df["is_fraud"].to_numpy(dtype=np.int32)
    x_eval = eval_df[feature_cols].to_numpy(dtype=np.float32)

    pos_count = max(1, int(np.sum(y_train == 1)))
    neg_count = max(1, int(np.sum(y_train == 0)))
    scale_pos_weight = float(neg_count / pos_count)

    clf = xgb.XGBClassifier(
        n_estimators=220 if include_graph_features else 180,
        max_depth=6 if include_graph_features else 5,
        learning_rate=0.06,
        subsample=0.85,
        colsample_bytree=0.85,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=seed,
    )
    clf.fit(x_train, y_train)

    probs = clf.predict_proba(x_eval)[:, 1]
    explainer = shap.TreeExplainer(clf)
    shap_vals = explainer.shap_values(x_eval)

    return XGBModelArtifact(
        name="XGBoost + Graph Features" if include_graph_features else "XGBoost (Tabular)",
        model=clf,
        feature_names=feature_cols,
        probs=probs,
        shap_values=np.asarray(shap_vals),
    )


def get_account_shap_explanation(
    artifact: XGBModelArtifact,
    eval_df: pd.DataFrame,
    account_id: str,
    top_k: int = 6,
) -> List[Dict[str, float | str]]:
    matches = eval_df.index[eval_df["account_id"] == account_id].tolist()
    if not matches:
        return []
    row_idx = matches[0]
    row_shap = artifact.shap_values[row_idx]
    order = np.argsort(np.abs(row_shap))[::-1][:top_k]

    explanations = []
    for feat_idx in order:
        feat_name = artifact.feature_names[feat_idx]
        explanations.append(
            {
                "feature": feat_name,
                "value": float(eval_df.iloc[row_idx][feat_name]),
                "shap_contribution": float(row_shap[feat_idx]),
            }
        )
    return explanations
