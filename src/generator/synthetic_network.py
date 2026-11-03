"""
Synthetic Transaction Network Generator with Planted Fraud Rings.

Generates accounts, devices, merchants, and timestamped transactions with
log-normal amount distributions and Poisson arrival times, then plants four
canonical fraud ring topologies:
  1. money_mule_chains: Sequential rapid multi-hop layering A -> B -> C -> D -> E
  2. shared_device_rings: Collusive accounts sharing device fingerprints & IP subnets
  3. burst_activity: High-velocity fan-out/fan-in attacks within tight time windows
  4. circular_transfers: Cyclic wash/round-trip transfers A -> B -> C -> D -> A

Supports post-training distribution shift (`shift_mode="tactic_shift"`) where
fraudsters rotate devices and throttle burst velocity in the test window.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple
import numpy as np
import pandas as pd
import networkx as nx


FraudPatternType = Literal[
    "legitimate",
    "money_mule_chain",
    "shared_device_ring",
    "burst_activity",
    "circular_transfer",
]


@dataclass
class GeneratorConfig:
    seed: int = 42
    num_accounts: int = 120
    num_devices: int = 95
    num_merchants: int = 30
    num_transactions: int = 850
    fraud_rate: float = 0.015  # 0.5% to 2.0% configurable
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

        # Assign baseline primary devices to accounts
        account_ids = accounts_df["account_id"].tolist()
        device_ids = devices_df["device_id"].tolist()
        primary_devices = {
            acc: device_ids[i % len(device_ids)]
            for i, acc in enumerate(account_ids)
        }

        # Generate legitimate baseline transactions
        tx_records: List[Dict] = []
        base_tx_count = int(self.cfg.num_transactions * (1.0 - self.cfg.fraud_rate * 4.5))
        base_tx_count = max(100, min(self.cfg.num_transactions - 20, base_tx_count))

        for tx_idx in range(base_tx_count):
            src, dst = self.rng.choice(account_ids, size=2, replace=False)
            ts = float(self.rng.uniform(0.0, self.cfg.time_span_hours * 0.92))
            amount = float(np.round(self.rng.lognormal(mean=4.1, sigma=0.85), 2))
            merch = str(self.rng.choice(merchants_df["merchant_id"]))
            tx_records.append(
                {
                    "tx_id": f"TX-{tx_idx:05d}",
                    "src_id": str(src),
                    "dst_id": str(dst),
                    "merchant_id": merch,
                    "device_id": primary_devices[src],
                    "timestamp_hr": round(ts, 4),
                    "amount": amount,
                    "is_fraud": 0,
                    "fraud_pattern": "legitimate",
                    "ring_id": "NONE",
                }
            )

        # Select disjoint account subsets for planted fraud patterns
        ring_memberships: Dict[str, List[str]] = {}
        fraud_accounts: Dict[str, Tuple[str, str]] = {}  # acc_id -> (pattern, ring_id)

        available_pool = list(account_ids[:])
        self.rng.shuffle(available_pool)

        # Scale number of fraud accounts according to fraud_rate
        target_fraud_accounts = max(
            12, int(round(self.cfg.num_accounts * self.cfg.fraud_rate * 8.0))
        )
        per_pattern_budget = max(3, target_fraud_accounts // 4)

        # 1. Plant Money-Mule Chains (early, mid, and test window)
        mule_nodes = available_pool[:per_pattern_budget]
        available_pool = available_pool[per_pattern_budget:]
        ring_id_mule = "RING-MULE-01"
        ring_memberships[ring_id_mule] = mule_nodes
        for acc in mule_nodes:
            fraud_accounts[acc] = ("money_mule_chain", ring_id_mule)

        for window_base in [28.0, 95.0, 148.0]:
            hop_amount = float(np.round(self.rng.uniform(1800.0, 3400.0), 2))
            for i in range(len(mule_nodes) - 1):
                src = mule_nodes[i]
                dst = mule_nodes[i + 1]
                in_test_shift = (
                    self.cfg.shift_mode == "tactic_shift"
                    and window_base > self.cfg.time_span_hours * self.cfg.train_cutoff_ratio
                )
                delay = 0.65 if in_test_shift else 0.15
                attenuation = 0.91 if in_test_shift else 0.97
                hop_amount = round(hop_amount * attenuation, 2)
                tx_records.append(
                    {
                        "tx_id": f"TX-MULE-{int(window_base)}-{i}",
                        "src_id": src,
                        "dst_id": dst,
                        "merchant_id": str(merchants_df["merchant_id"].iloc[0]),
                        "device_id": primary_devices[src],
                        "timestamp_hr": round(window_base + i * delay, 4),
                        "amount": hop_amount,
                        "is_fraud": 1,
                        "fraud_pattern": "money_mule_chain",
                        "ring_id": ring_id_mule,
                    }
                )

        # 2. Plant Shared-Device Rings
        sd_nodes = available_pool[:per_pattern_budget]
        available_pool = available_pool[per_pattern_budget:]
        ring_id_sd = "RING-SHARED-02"
        ring_memberships[ring_id_sd] = sd_nodes
        shared_dev_id = device_ids[0]
        alt_dev_id = device_ids[1]
        for acc in sd_nodes:
            fraud_accounts[acc] = ("shared_device_ring", ring_id_sd)
            primary_devices[acc] = shared_dev_id

        for window_base in [42.0, 112.0, 152.0]:
            in_test_shift = (
                self.cfg.shift_mode == "tactic_shift"
                and window_base > self.cfg.time_span_hours * self.cfg.train_cutoff_ratio
            )
            for i, src in enumerate(sd_nodes):
                dst = sd_nodes[(i + 1) % len(sd_nodes)]
                used_device = (
                    alt_dev_id if (in_test_shift and i % 2 == 1) else shared_dev_id
                )
                tx_records.append(
                    {
                        "tx_id": f"TX-SDEV-{int(window_base)}-{i}",
                        "src_id": src,
                        "dst_id": dst,
                        "merchant_id": str(merchants_df["merchant_id"].iloc[1]),
                        "device_id": used_device,
                        "timestamp_hr": round(window_base + i * 0.08, 4),
                        "amount": float(np.round(self.rng.uniform(750.0, 1650.0), 2)),
                        "is_fraud": 1,
                        "fraud_pattern": "shared_device_ring",
                        "ring_id": ring_id_sd,
                    }
                )

        # 3. Plant Burst Activity Clusters
        burst_nodes = available_pool[:per_pattern_budget]
        available_pool = available_pool[per_pattern_budget:]
        ring_id_burst = "RING-BURST-03"
        ring_memberships[ring_id_burst] = burst_nodes
        for acc in burst_nodes:
            fraud_accounts[acc] = ("burst_activity", ring_id_burst)

        hub_acc = burst_nodes[0]
        spokes = burst_nodes[1:] if len(burst_nodes) > 1 else [account_ids[0]]
        for window_base in [54.0, 120.0, 156.0]:
            in_test_shift = (
                self.cfg.shift_mode == "tactic_shift"
                and window_base > self.cfg.time_span_hours * self.cfg.train_cutoff_ratio
            )
            burst_count = 6 if in_test_shift else 10
            step_hr = 0.22 if in_test_shift else 0.03
            for b_idx in range(burst_count):
                spoke = spokes[b_idx % len(spokes)]
                tx_records.append(
                    {
                        "tx_id": f"TX-BURST-{int(window_base)}-{b_idx}",
                        "src_id": hub_acc,
                        "dst_id": spoke,
                        "merchant_id": str(merchants_df["merchant_id"].iloc[2]),
                        "device_id": primary_devices[hub_acc],
                        "timestamp_hr": round(window_base + b_idx * step_hr, 4),
                        "amount": float(
                            np.round(
                                self.rng.uniform(320.0, 620.0)
                                if in_test_shift
                                else self.rng.uniform(950.0, 1490.0),
                                2,
                            )
                        ),
                        "is_fraud": 1,
                        "fraud_pattern": "burst_activity",
                        "ring_id": ring_id_burst,
                    }
                )

        # 4. Plant Circular Transfers
        cycle_nodes = available_pool[:per_pattern_budget]
        ring_id_cycle = "RING-CYCLE-04"
        ring_memberships[ring_id_cycle] = cycle_nodes
        for acc in cycle_nodes:
            fraud_accounts[acc] = ("circular_transfer", ring_id_cycle)

        for window_base in [64.0, 126.0, 160.0]:
            base_cycle_amt = float(np.round(self.rng.uniform(2400.0, 4200.0), 2))
            for i in range(len(cycle_nodes)):
                src = cycle_nodes[i]
                dst = cycle_nodes[(i + 1) % len(cycle_nodes)]
                tx_records.append(
                    {
                        "tx_id": f"TX-CYCLE-{int(window_base)}-{i}",
                        "src_id": src,
                        "dst_id": dst,
                        "merchant_id": str(merchants_df["merchant_id"].iloc[3]),
                        "device_id": primary_devices[src],
                        "timestamp_hr": round(window_base + i * 0.11, 4),
                        "amount": round(base_cycle_amt * (0.985 ** i), 2),
                        "is_fraud": 1,
                        "fraud_pattern": "circular_transfer",
                        "ring_id": ring_id_cycle,
                    }
                )

        # Annotate accounts dataframe with planted ground-truth labels
        accounts_df["primary_device_id"] = accounts_df["account_id"].map(primary_devices)
        accounts_df["is_fraud"] = accounts_df["account_id"].map(
            lambda x: 1 if x in fraud_accounts else 0
        )
        accounts_df["fraud_pattern"] = accounts_df["account_id"].map(
            lambda x: fraud_accounts[x][0] if x in fraud_accounts else "legitimate"
        )
        accounts_df["ring_id"] = accounts_df["account_id"].map(
            lambda x: fraud_accounts[x][1] if x in fraud_accounts else "NONE"
        )

        transactions_df = (
            pd.DataFrame(tx_records)
            .sort_values("timestamp_hr", kind="mergesort")
            .reset_index(drop=True)
        )

        graph = self._build_networkx_graph(accounts_df, transactions_df)
        return SyntheticDataset(
            accounts_df=accounts_df,
            devices_df=devices_df,
            merchants_df=merchants_df,
            transactions_df=transactions_df,
            graph=graph,
            ring_memberships=ring_memberships,
        )

    def _generate_accounts(self) -> pd.DataFrame:
        rows = []
        regions = ["US-EAST", "US-WEST", "EU-CENTRAL", "UK-LON", "APAC-SG"]
        for i in range(self.cfg.num_accounts):
            rows.append(
                {
                    "account_id": f"ACC-{1000 + i}",
                    "account_age_days": int(self.rng.integers(5, 1400)),
                    "kyc_tier": int(self.rng.choice([1, 2, 3], p=[0.25, 0.55, 0.20])),
                    "region": str(self.rng.choice(regions)),
                    "declared_monthly_vol": float(
                        np.round(self.rng.lognormal(mean=7.5, sigma=0.7), 2)
                    ),
                }
            )
        return pd.DataFrame(rows)

    def _generate_devices(self) -> pd.DataFrame:
        rows = []
        os_families = ["iOS 17", "Android 14", "macOS Sonoma", "Windows 11", "Linux x64"]
        for i in range(self.cfg.num_devices):
            rows.append(
                {
                    "device_id": f"DEV-{200 + i:03d}",
                    "os_family": str(self.rng.choice(os_families)),
                    "is_emulator": int(self.rng.choice([0, 1], p=[0.92, 0.08])),
                    "ip_subnet": f"192.0.{int(self.rng.integers(10, 250))}.0/24",
                }
            )
        return pd.DataFrame(rows)

    def _generate_merchants(self) -> pd.DataFrame:
        rows = []
        categories = ["wire_transfer", "crypto_onramp", "p2p_wallet", "retail_pos", "fx_desk"]
        for i in range(self.cfg.num_merchants):
            rows.append(
                {
                    "merchant_id": f"MER-{300 + i:03d}",
                    "category": categories[i % len(categories)],
                    "risk_tier": int(self.rng.choice([1, 2, 3], p=[0.6, 0.3, 0.1])),
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def _build_networkx_graph(
        accounts_df: pd.DataFrame, transactions_df: pd.DataFrame
    ) -> nx.MultiDiGraph:
        g = nx.MultiDiGraph()
        for _, row in accounts_df.iterrows():
            g.add_node(row["account_id"], **row.to_dict())
        for _, tx in transactions_df.iterrows():
            g.add_edge(tx["src_id"], tx["dst_id"], **tx.to_dict())
        return g
