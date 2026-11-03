"""
Feature Engineering Pipeline (Tabular vs Hand-Crafted Graph Features)
with strict temporal cutoff enforcement to prevent data leakage.
"""

from __future__ import annotations

from typing import List, Tuple
import numpy as np
import pandas as pd
import networkx as nx


TABULAR_FEATURE_COLUMNS: List[str] = [
    "account_age_days",
    "kyc_tier",
    "tx_sent_count",
    "tx_recv_count",
    "sent_amount_mean",
    "sent_amount_max",
    "recv_amount_mean",
    "burst_tx_1h_max",
    "amt_to_declared_ratio",
]

GRAPH_FEATURE_COLUMNS: List[str] = [
    "in_degree",
    "out_degree",
    "pagerank",
    "shared_device_accounts",
    "clustering_coeff",
    "cycle_participation",
    "two_hop_high_velocity_neighbors",
]


def build_features_up_to_time(
    accounts_df: pd.DataFrame,
    transactions_df: pd.DataFrame,
    max_timestamp_hr: float,
) -> pd.DataFrame:
    """
    Computes tabular and graph topology features strictly using transactions with
    `timestamp_hr <= max_timestamp_hr`. Guarantees zero future leakage.
    """
    observed_tx = transactions_df[
        transactions_df["timestamp_hr"] <= max_timestamp_hr
    ].copy()

    # Build directed graph strictly from observed window
    dg = nx.DiGraph()
    for acc_id in accounts_df["account_id"]:
        dg.add_node(acc_id)

    for _, tx in observed_tx.iterrows():
        u, v = tx["src_id"], tx["dst_id"]
        if dg.has_edge(u, v):
            dg[u][v]["weight"] += float(tx["amount"])
            dg[u][v]["count"] += 1
        else:
            dg.add_edge(u, v, weight=float(tx["amount"]), count=1)

    pagerank_map = nx.pagerank(dg, alpha=0.85) if dg.number_of_edges() > 0 else {
        n: 1.0 / max(1, dg.number_of_nodes()) for n in dg.nodes()
    }
    undirected_g = dg.to_undirected()
    clustering_map = nx.clustering(undirected_g)

    # Device sharing map within observed transactions
    dev_to_accounts = (
        observed_tx.groupby("device_id")["src_id"].nunique().to_dict()
        if not observed_tx.empty
        else {}
    )

    # Cycle participation (bound length <= 4)
    cycle_counts = {acc: 0 for acc in accounts_df["account_id"]}
    for cycle in nx.simple_cycles(dg, length_bound=4):
        for node in cycle:
            cycle_counts[node] = cycle_counts.get(node, 0) + 1

    feature_rows = []
    for _, acc in accounts_df.iterrows():
        acc_id = acc["account_id"]
        sent = observed_tx[observed_tx["src_id"] == acc_id]
        recv = observed_tx[observed_tx["dst_id"] == acc_id]

        # Max burst within any 1-hour sliding window
        burst_max = 0
        if not sent.empty:
            ts_sorted = np.sort(sent["timestamp_hr"].to_numpy())
            left = 0
            for right in range(len(ts_sorted)):
                while ts_sorted[right] - ts_sorted[left] > 1.0:
                    left += 1
                burst_max = max(burst_max, right - left + 1)

        # Shared device degree
        used_devices = sent["device_id"].unique().tolist()
        shared_dev_count = max(
            [dev_to_accounts.get(d, 1) for d in used_devices],
            default=1,
        )

        sent_mean = float(sent["amount"].mean()) if not sent.empty else 0.0
        sent_max = float(sent["amount"].max()) if not sent.empty else 0.0
        recv_mean = float(recv["amount"].mean()) if not recv.empty else 0.0

        # 2-hop high velocity neighbors
        nbrs_1 = set(dg.successors(acc_id)) | set(dg.predecessors(acc_id))
        nbrs_2 = set()
        for n1 in nbrs_1:
            nbrs_2 |= set(dg.successors(n1)) | set(dg.predecessors(n1))
        nbrs_2.discard(acc_id)
        high_vel_2hop = sum(1 for n2 in nbrs_2 if dg.degree(n2) >= 5)

        feature_rows.append(
            {
                "account_id": acc_id,
                "account_age_days": float(acc["account_age_days"]),
                "kyc_tier": float(acc["kyc_tier"]),
                "tx_sent_count": float(len(sent)),
                "tx_recv_count": float(len(recv)),
                "sent_amount_mean": round(sent_mean, 4),
                "sent_amount_max": round(sent_max, 4),
                "recv_amount_mean": round(recv_mean, 4),
                "burst_tx_1h_max": float(burst_max),
                "amt_to_declared_ratio": round(
                    sent_max / max(100.0, float(acc["declared_monthly_vol"])), 4
                ),
                # Hand-crafted graph features
                "in_degree": float(dg.in_degree(acc_id)),
                "out_degree": float(dg.out_degree(acc_id)),
                "pagerank": round(float(pagerank_map.get(acc_id, 0.0)), 6),
                "shared_device_accounts": float(shared_dev_count),
                "clustering_coeff": round(float(clustering_map.get(acc_id, 0.0)), 4),
                "cycle_participation": float(cycle_counts.get(acc_id, 0)),
                "two_hop_high_velocity_neighbors": float(high_vel_2hop),
                "is_fraud": int(acc["is_fraud"]),
                "fraud_pattern": str(acc["fraud_pattern"]),
                "ring_id": str(acc["ring_id"]),
            }
        )

    return pd.DataFrame(feature_rows)


def temporal_train_val_test_split(
    transactions_df: pd.DataFrame,
    train_ratio: float = 0.60,
    val_ratio: float = 0.20,
) -> Tuple[float, float, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits transactions strictly by timestamp_hr without shuffling across windows.
    Returns (train_cutoff_hr, val_cutoff_hr, train_tx, val_tx, test_tx).
    """
    sorted_ts = np.sort(transactions_df["timestamp_hr"].to_numpy())
    train_cutoff = float(np.quantile(sorted_ts, train_ratio))
    val_cutoff = float(np.quantile(sorted_ts, train_ratio + val_ratio))

    train_tx = transactions_df[transactions_df["timestamp_hr"] <= train_cutoff].copy()
    val_tx = transactions_df[
        (transactions_df["timestamp_hr"] > train_cutoff)
        & (transactions_df["timestamp_hr"] <= val_cutoff)
    ].copy()
    test_tx = transactions_df[transactions_df["timestamp_hr"] > val_cutoff].copy()
    return train_cutoff, val_cutoff, train_tx, val_tx, test_tx
