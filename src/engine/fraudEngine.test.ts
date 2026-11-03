import test from 'node:test';
import assert from 'node:assert/strict';
import {
  runFraudRingSimulation,
  extractAccountFeaturesUpToTime,
  computeAccountShapExplanation,
  computeGnnExplainerSubgraph,
  GeneratorParams,
} from './fraudEngine';

const defaultParams: GeneratorParams = {
  seed: 42,
  numAccounts: 96,
  numTransactions: 520,
  fraudRate: 0.015,
  shiftMode: 'none',
  trainRatio: 0.6,
  valRatio: 0.2,
  fixedPrecisionTarget: 0.8,
};

test('generator is strictly deterministic for identical seeds', () => {
  const simA = runFraudRingSimulation(defaultParams);
  const simB = runFraudRingSimulation({ ...defaultParams });

  assert.equal(simA.accounts.length, simB.accounts.length);
  assert.equal(simA.transactions.length, simB.transactions.length);
  assert.deepEqual(
    simA.accounts.map((a) => a.id),
    simB.accounts.map((a) => a.id)
  );
  assert.deepEqual(
    simA.transactions.map((t) => t.amount),
    simB.transactions.map((t) => t.amount)
  );
  assert.deepEqual(simA.rings, simB.rings);
});

test('different seeds produce distinct transaction amounts and layouts', () => {
  const simA = runFraudRingSimulation({ ...defaultParams, seed: 42 });
  const simB = runFraudRingSimulation({ ...defaultParams, seed: 99 });

  assert.notDeepEqual(
    simA.transactions.slice(0, 15).map((t) => t.amount),
    simB.transactions.slice(0, 15).map((t) => t.amount)
  );
});

test('strict temporal split has zero overlap and zero future data leakage', () => {
  const sim = runFraudRingSimulation(defaultParams);
  assert.equal(sim.leakageCheckPassed, true);

  const trainTxs = sim.transactions.filter((t) => t.splitWindow === 'train');
  const valTxs = sim.transactions.filter((t) => t.splitWindow === 'val');
  const testTxs = sim.transactions.filter((t) => t.splitWindow === 'test');

  const maxTrainTs = Math.max(...trainTxs.map((t) => t.timestampHr));
  const minValTs = Math.min(...valTxs.map((t) => t.timestampHr));
  const maxValTs = Math.max(...valTxs.map((t) => t.timestampHr));
  const minTestTs = Math.min(...testTxs.map((t) => t.timestampHr));

  assert.ok(maxTrainTs <= sim.trainCutoffHr);
  assert.ok(minValTs >= maxTrainTs);
  assert.ok(maxValTs <= sim.valCutoffHr);
  assert.ok(minTestTs >= maxValTs);

  // Verify feature extraction invariance when future transactions are corrupted
  const cleanTrainFeats = extractAccountFeaturesUpToTime(
    sim.accounts,
    sim.transactions,
    sim.trainCutoffHr
  );
  const corruptedFuture = sim.transactions.map((t) =>
    t.timestampHr > sim.trainCutoffHr ? { ...t, amount: 9999999, deviceId: 'DEV-HACK' } : t
  );
  const afterCorruptionFeats = extractAccountFeaturesUpToTime(
    sim.accounts,
    corruptedFuture,
    sim.trainCutoffHr
  );

  assert.deepEqual(cleanTrainFeats, afterCorruptionFeats);
});

test('all four planted fraud patterns exist and GNN models outperform tabular baseline on topological rings', () => {
  const sim = runFraudRingSimulation(defaultParams);
  const ringPatterns = sim.rings.map((r) => r.pattern).sort();
  assert.deepEqual(ringPatterns, [
    'burst_activity',
    'circular_transfer',
    'money_mule_chain',
    'shared_device_ring',
  ]);

  const tabPrAuc = sim.evaluations.xgboost_tabular.prAuc;
  const graphPrAuc = sim.evaluations.xgboost_graph.prAuc;
  const sagePrAuc = sim.evaluations.graphsage_focal.prAuc;
  const gatPrAuc = sim.evaluations.gat_weighted.prAuc;

  assert.ok(
    graphPrAuc > tabPrAuc,
    `Expected XGBoost+Graph (${graphPrAuc}) > XGBoost Tabular (${tabPrAuc})`
  );
  assert.ok(
    sagePrAuc > tabPrAuc,
    `Expected GraphSAGE (${sagePrAuc}) > XGBoost Tabular (${tabPrAuc})`
  );
  assert.ok(
    gatPrAuc > tabPrAuc,
    `Expected GAT (${gatPrAuc}) > XGBoost Tabular (${tabPrAuc})`
  );
});

test('SHAP and GNNExplainer return valid ordered attributions for flagged accounts', () => {
  const sim = runFraudRingSimulation(defaultParams);
  const fraudAccount = sim.accounts.find((a) => a.isFraud)!;
  const feat = sim.featuresByAccount[fraudAccount.id];

  const shapItems = computeAccountShapExplanation(feat, true);
  assert.ok(shapItems.length >= 6);
  assert.ok(Math.abs(shapItems[0].shapDelta) >= Math.abs(shapItems[1].shapDelta));

  const subgraph = computeGnnExplainerSubgraph(
    fraudAccount.id,
    sim.accounts,
    sim.transactions
  );
  assert.ok(subgraph.length > 0);
  assert.ok(subgraph[0].importanceWeight >= subgraph[subgraph.length - 1].importanceWeight);
});
