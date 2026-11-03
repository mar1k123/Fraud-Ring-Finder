"""
Unit Tests for Fraud Ring Finder:
  1. Generator Determinism: identical seeds yield byte-identical accounts,
     transactions, and planted fraud ring memberships.
  2. Strict Temporal Split & Zero Data Leakage: feature computation up to `t_cutoff`
     is invariant to any mutations or additions in future transactions (`t > t_cutoff`).
"""

from __future__ import annotations

import unittest
import pandas as pd

from src.generator.synthetic_network import FraudNetworkGenerator, GeneratorConfig
from src.features.graph_features import (
    build_features_up_to_time,
    temporal_train_val_test_split,
)


class TestGeneratorDeterminismAndLeakage(unittest.TestCase):
    def test_generator_is_strictly_deterministic(self) -> None:
        cfg1 = GeneratorConfig(seed=1337, num_accounts=80, num_transactions=400, fraud_rate=0.015)
        cfg2 = GeneratorConfig(seed=1337, num_accounts=80, num_transactions=400, fraud_rate=0.015)

        ds1 = FraudNetworkGenerator(cfg1).generate()
        ds2 = FraudNetworkGenerator(cfg2).generate()

        pd.testing.assert_frame_equal(ds1.accounts_df, ds2.accounts_df)
        pd.testing.assert_frame_equal(ds1.transactions_df, ds2.transactions_df)
        self.assertEqual(ds1.ring_memberships, ds2.ring_memberships)

    def test_different_seeds_produce_distinct_networks(self) -> None:
        ds1 = FraudNetworkGenerator(GeneratorConfig(seed=101)).generate()
        ds2 = FraudNetworkGenerator(GeneratorConfig(seed=202)).generate()
        self.assertFalse(ds1.transactions_df["amount"].equals(ds2.transactions_df["amount"]))

    def test_temporal_split_has_zero_overlap(self) -> None:
        ds = FraudNetworkGenerator(GeneratorConfig(seed=42)).generate()
        train_cut, val_cut, train_tx, val_tx, test_tx = temporal_train_val_test_split(
            ds.transactions_df, train_ratio=0.60, val_ratio=0.20
        )
        self.assertLessEqual(train_tx["timestamp_hr"].max(), train_cut)
        self.assertGreater(val_tx["timestamp_hr"].min(), train_cut)
        self.assertLessEqual(val_tx["timestamp_hr"].max(), val_cut)
        self.assertGreater(test_tx["timestamp_hr"].min(), val_cut)

    def test_no_future_leakage_in_feature_extraction(self) -> None:
        ds = FraudNetworkGenerator(GeneratorConfig(seed=42)).generate()
        train_cut, _, _, _, _ = temporal_train_val_test_split(ds.transactions_df)

        features_clean = build_features_up_to_time(
            ds.accounts_df, ds.transactions_df, max_timestamp_hr=train_cut
        )

        # Corrupt all future transactions (timestamp_hr > train_cut) with extreme amounts
        corrupted_tx = ds.transactions_df.copy()
        future_mask = corrupted_tx["timestamp_hr"] > train_cut
        corrupted_tx.loc[future_mask, "amount"] = 9999999.0

        features_after_future_mutation = build_features_up_to_time(
            ds.accounts_df, corrupted_tx, max_timestamp_hr=train_cut
        )

        pd.testing.assert_frame_equal(features_clean, features_after_future_mutation)


if __name__ == "__main__":
    unittest.main()
