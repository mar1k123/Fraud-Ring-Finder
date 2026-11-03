"""
Streamlit + PyVis Interactive Workbench for Fraud Ring Finder.
Run locally via:
    streamlit run src/app_streamlit.py
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import streamlit.components.v1 as components
from pyvis.network import Network

from src.generator.synthetic_network import FraudNetworkGenerator, GeneratorConfig
from src.features.graph_features import (
    build_features_up_to_time,
    temporal_train_val_test_split,
)
from src.models.baselines import train_xgboost_baseline, get_account_shap_explanation
from src.evaluation.metrics import evaluate_model_predictions


st.set_page_config(
    page_title="Fraud Ring Finder — Graph ML Workbench",
    layout="wide",
)

st.title("Fraud Ring Finder — Graph ML Transaction Forensics")
st.caption(
    "Synthetic Transaction Network Generator · XGBoost + Graph Features · GraphSAGE & GAT · SHAP & GNNExplainer"
)

with st.sidebar:
    st.header("Generator & Model Controls")
    seed = st.number_input("Random Seed", min_value=1, max_value=99999, value=42)
    fraud_rate = st.slider(
        "Target Fraud Rate", min_value=0.005, max_value=0.025, value=0.015, step=0.001
    )
    shift_mode = st.selectbox(
        "Post-Training Distribution Shift",
        options=["none", "tactic_shift"],
        help="Simulate fraudsters rotating devices and throttling burst velocity after the training window.",
    )
    target_precision = st.slider(
        "Fixed Precision Target", min_value=0.60, max_value=0.95, value=0.80, step=0.05
    )

cfg = GeneratorConfig(
    seed=int(seed),
    fraud_rate=float(fraud_rate),
    shift_mode=shift_mode,
)
dataset = FraudNetworkGenerator(cfg).generate()

train_cut, val_cut, train_tx, val_tx, test_tx = temporal_train_val_test_split(
    dataset.transactions_df
)
train_feat = build_features_up_to_time(
    dataset.accounts_df, dataset.transactions_df, train_cut
)
eval_feat = build_features_up_to_time(
    dataset.accounts_df, dataset.transactions_df, dataset.cfg.time_span_hours
)

xgb_tab = train_xgboost_baseline(train_feat, eval_feat, include_graph_features=False, seed=seed)
xgb_graph = train_xgboost_baseline(train_feat, eval_feat, include_graph_features=True, seed=seed)

eval_tab = evaluate_model_predictions(
    "XGBoost (Tabular)", eval_feat, xgb_tab.probs, target_precision
)
eval_graph = evaluate_model_predictions(
    "XGBoost + Graph Features", eval_feat, xgb_graph.probs, target_precision
)

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Interactive Transaction Network (PyVis)")
    net = Network(height="580px", width="100%", bgcolor="#0F172A", font_color="#F8FAFC", directed=True)
    for idx, row in eval_feat.iterrows():
        acc_id = row["account_id"]
        risk = float(xgb_graph.probs[idx])
        color = "#DC2626" if row["is_fraud"] == 1 else ("#D97706" if risk > 0.45 else "#475569")
        net.add_node(
            acc_id,
            label=acc_id,
            title=f"{acc_id} | Pattern: {row['fraud_pattern']} | Risk: {risk:.3f}",
            color=color,
            size=22 if row["is_fraud"] == 1 else 12,
        )

    for _, tx in dataset.transactions_df.tail(260).iterrows():
        edge_color = "#EF4444" if tx["is_fraud"] == 1 else "#334155"
        net.add_edge(tx["src_id"], tx["dst_id"], color=edge_color, value=float(tx["amount"]))

    html_content = net.generate_html()
    components.html(html_content, height=600)

with col2:
    st.subheader("Per-Account Risk Card & SHAP Attribution")
    flagged_accounts = eval_feat.sort_values("is_fraud", ascending=False)["account_id"].tolist()
    selected_acc = st.selectbox("Select Account", options=flagged_accounts)
    acc_row = eval_feat[eval_feat["account_id"] == selected_acc].iloc[0]
    acc_idx = eval_feat.index[eval_feat["account_id"] == selected_acc][0]

    st.metric("Predicted Fraud Risk", f"{xgb_graph.probs[acc_idx]:.3f}")
    st.write(f"**Ring ID:** `{acc_row['ring_id']}` · **Pattern:** `{acc_row['fraud_pattern']}`")
    shap_items = get_account_shap_explanation(xgb_graph, eval_feat, selected_acc, top_k=6)
    st.dataframe(pd.DataFrame(shap_items), use_container_width=True)
