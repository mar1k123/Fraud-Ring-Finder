/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useMemo } from 'react';
import {
  Network,
  BarChart3,
  ShieldAlert,
  FileCode2,
  Sliders,
  RefreshCw,
  Download,
} from 'lucide-react';
import {
  GeneratorParams,
  FraudPattern,
  ModelId,
  ShiftMode,
  runFraudRingSimulation,
  computeAccountShapExplanation,
  computeGnnExplainerSubgraph,
} from './engine/fraudEngine';
import { NetworkGraphCanvas } from './components/NetworkGraphCanvas';
import { AccountRiskInspector } from './components/AccountRiskInspector';
import { EvaluationAblationView } from './components/EvaluationAblationView';
import { CodeAndReadmeExplorer } from './components/CodeAndReadmeExplorer';

type WorkspaceTab = 'workbench' | 'evaluation' | 'rings' | 'repository';

export default function App() {
  const [activeTab, setActiveTab] = useState<WorkspaceTab>('workbench');
  const [params, setParams] = useState<GeneratorParams>({
    seed: 42,
    numAccounts: 96,
    numTransactions: 540,
    fraudRate: 0.015,
    shiftMode: 'none',
    trainRatio: 0.6,
    valRatio: 0.2,
    fixedPrecisionTarget: 0.8,
  });

  const [activeModelId, setActiveModelId] = useState<ModelId>('gat_weighted');
  const [patternFilter, setPatternFilter] = useState<FraudPattern | 'all'>('all');
  const [showOnlyFlaggedRings, setShowOnlyFlaggedRings] = useState<boolean>(false);

  const simulation = useMemo(() => runFraudRingSimulation(params), [params]);

  const [selectedAccountId, setSelectedAccountId] = useState<string>('ACC-1000');

  const selectedAccount = useMemo(() => {
    return (
      simulation.accounts.find((a) => a.id === selectedAccountId) ||
      simulation.accounts[0]
    );
  }, [simulation.accounts, selectedAccountId]);

  const selectedFeatures =
    simulation.featuresByAccount[selectedAccount.id] ||
    simulation.featuresByAccount[simulation.accounts[0].id];

  const activeEval = simulation.evaluations[activeModelId];

  const shapAttributions = useMemo(() => {
    const usesGraph = activeModelId !== 'xgboost_tabular';
    return computeAccountShapExplanation(selectedFeatures, usesGraph);
  }, [selectedFeatures, activeModelId]);

  const gnnSubgraph = useMemo(() => {
    return computeGnnExplainerSubgraph(
      selectedAccount.id,
      simulation.accounts,
      simulation.transactions
    );
  }, [selectedAccount.id, simulation.accounts, simulation.transactions]);

  const handleExportReportJson = () => {
    const payload = {
      project: 'fraud-ring-finder',
      generatedAt: new Date().toISOString(),
      params: simulation.params,
      temporalSplit: {
        trainCutoffHr: simulation.trainCutoffHr,
        valCutoffHr: simulation.valCutoffHr,
        leakageCheckPassed: simulation.leakageCheckPassed,
      },
      rings: simulation.rings,
      modelBenchmarks: Object.values(simulation.evaluations).map((ev) => ({
        modelId: ev.modelId,
        modelName: ev.modelName,
        prAuc: ev.prAuc,
        recallAtFixedPrecision: ev.recallAtFixedPrecision,
        patternRecall: ev.patternRecall,
        preShiftPrAuc: ev.preShiftPrAuc,
        postShiftPrAuc: ev.postShiftPrAuc,
      })),
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], {
      type: 'application/json',
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `fraud-ring-finder-seed-${params.seed}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleShiftToggle = (mode: ShiftMode) => {
    setParams((prev) => ({ ...prev, shiftMode: mode }));
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col">
      {/* Top Bar Contract: Zone 1 Brand Wordmark — Zone 2 Nav Links — Zone 3 Primary Action */}
      <header className="flex items-center justify-between gap-8 px-6 py-3.5 border-b border-slate-800 bg-slate-900">
        <a
          href="#top"
          onClick={(e) => {
            e.preventDefault();
            setActiveTab('workbench');
          }}
          className="text-lg font-bold tracking-tight text-white whitespace-nowrap shrink-0 font-display"
        >
          Fraud Ring Finder
        </a>

        <nav
          aria-label="Primary Workspace Navigation"
          className="flex items-center gap-6 text-sm font-medium text-slate-400"
        >
          <button
            type="button"
            onClick={() => setActiveTab('workbench')}
            className={`hover:text-white transition-colors whitespace-nowrap shrink-0 pb-0.5 ${
              activeTab === 'workbench'
                ? 'text-white border-b-2 border-amber-400'
                : ''
            }`}
          >
            Network Workbench
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('evaluation')}
            className={`hover:text-white transition-colors whitespace-nowrap shrink-0 pb-0.5 ${
              activeTab === 'evaluation'
                ? 'text-white border-b-2 border-amber-400'
                : ''
            }`}
          >
            Evaluation & Ablation
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('rings')}
            className={`hover:text-white transition-colors whitespace-nowrap shrink-0 pb-0.5 ${
              activeTab === 'rings'
                ? 'text-white border-b-2 border-amber-400'
                : ''
            }`}
          >
            Planted Rings ({simulation.rings.length})
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('repository')}
            className={`hover:text-white transition-colors whitespace-nowrap shrink-0 pb-0.5 ${
              activeTab === 'repository'
                ? 'text-white border-b-2 border-amber-400'
                : ''
            }`}
          >
            Python Source & README
          </button>
        </nav>

        <div className="flex items-center gap-3 shrink-0">
          <button
            type="button"
            onClick={handleExportReportJson}
            className="flex items-center gap-1.5 px-3.5 py-1.5 text-xs font-semibold text-slate-950 bg-amber-400 hover:bg-amber-300 rounded-md transition-colors whitespace-nowrap shrink-0"
          >
            <Download className="w-3.5 h-3.5" />
            Export Benchmark JSON
          </button>
        </div>
      </header>

      {/* Desktop Control Strip: Seeded Generator, Fraud Rate, Active Model, Shift Mode */}
      <section
        aria-label="Synthetic Generator and Model Controls"
        className="border-b border-slate-800 bg-slate-900/60 px-6 py-3"
      >
        <div className="max-w-[1440px] mx-auto flex flex-wrap items-center justify-between gap-4">
          <div className="flex flex-wrap items-center gap-4 text-xs">
            <div className="flex items-center gap-2">
              <Sliders className="w-3.5 h-3.5 text-amber-400" />
              <label htmlFor="seed-input" className="text-slate-400 font-medium">
                Seed:
              </label>
              <input
                id="seed-input"
                type="number"
                min={1}
                max={99999}
                value={params.seed}
                onChange={(e) =>
                  setParams((p) => ({
                    ...p,
                    seed: Math.max(1, parseInt(e.target.value || '42', 10)),
                  }))
                }
                className="w-20 bg-slate-950 border border-slate-700 rounded px-2 py-1 text-slate-100 font-mono tabular-nums focus:outline-none focus:border-amber-400"
              />
              <button
                type="button"
                onClick={() =>
                  setParams((p) => ({ ...p, seed: ((p.seed * 31 + 7) % 9000) + 10 }))
                }
                className="flex items-center gap-1 px-2.5 py-1 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded transition-colors whitespace-nowrap shrink-0"
              >
                <RefreshCw className="w-3 h-3" />
                Reseed Graph
              </button>
            </div>

            <div className="flex items-center gap-2">
              <label htmlFor="fraud-rate-slider" className="text-slate-400 font-medium">
                Planted Fraud Rate:
              </label>
              <input
                id="fraud-rate-slider"
                type="range"
                min={0.005}
                max={0.02}
                step={0.0025}
                value={params.fraudRate}
                onChange={(e) =>
                  setParams((p) => ({ ...p, fraudRate: parseFloat(e.target.value) }))
                }
                className="w-24 accent-amber-400"
              />
              <span className="font-mono tabular-nums text-amber-300 font-semibold">
                {(params.fraudRate * 100).toFixed(2)}%
              </span>
            </div>

            <div className="flex items-center gap-2">
              <label htmlFor="model-selector" className="text-slate-400 font-medium">
                Active Model:
              </label>
              <select
                id="model-selector"
                value={activeModelId}
                onChange={(e) => setActiveModelId(e.target.value as ModelId)}
                className="bg-slate-950 border border-slate-700 rounded px-2.5 py-1 text-slate-100 font-medium focus:outline-none focus:border-amber-400"
              >
                <option value="xgboost_tabular">XGBoost (Tabular Baseline)</option>
                <option value="xgboost_graph">XGBoost + Graph Features</option>
                <option value="graphsage_focal">GraphSAGE (Focal Loss α=0.85)</option>
                <option value="gat_weighted">GAT (4-Head Weighted CE)</option>
              </select>
            </div>

            <div className="flex items-center gap-2">
              <span className="text-slate-400 font-medium">Test Regime:</span>
              <button
                type="button"
                onClick={() =>
                  handleShiftToggle(
                    params.shiftMode === 'none' ? 'tactic_shift' : 'none'
                  )
                }
                className={`px-2.5 py-1 rounded font-medium border transition-colors whitespace-nowrap shrink-0 ${
                  params.shiftMode === 'tactic_shift'
                    ? 'bg-amber-500/20 border-amber-500/50 text-amber-300'
                    : 'bg-slate-950 border-slate-700 text-slate-300 hover:text-white'
                }`}
              >
                {params.shiftMode === 'tactic_shift'
                  ? 'Distribution Shift: Active'
                  : 'Stationary (No Shift)'}
              </button>
            </div>
          </div>

          {/* Live Telemetry Summary Strip */}
          <div className="flex items-center gap-5 text-xs font-mono tabular-nums text-slate-300">
            <div>
              PR-AUC:{' '}
              <strong className="text-emerald-400">
                {activeEval.prAuc.toFixed(4)}
              </strong>
            </div>
            <div>
              Recall@80%Prec:{' '}
              <strong className="text-amber-300">
                {(activeEval.recallAtFixedPrecision * 100).toFixed(1)}%
              </strong>
            </div>
            <div>
              No-Leakage Split:{' '}
              <strong className="text-sky-400">
                {simulation.leakageCheckPassed ? 'PASS' : 'FAIL'}
              </strong>
            </div>
          </div>
        </div>
      </section>

      {/* Main Content Viewport */}
      <main className="flex-1 max-w-[1440px] w-full mx-auto px-6 py-5">
        {activeTab === 'workbench' && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-start">
            {/* Left 8 Columns: Interactive PyVis Network Topology + Quick Ring Cards */}
            <div className="lg:col-span-8 space-y-4">
              <NetworkGraphCanvas
                accounts={simulation.accounts}
                transactions={simulation.transactions}
                activeEval={activeEval}
                selectedAccountId={selectedAccount.id}
                onSelectAccount={(id) => setSelectedAccountId(id)}
                patternFilter={patternFilter}
                onChangePatternFilter={setPatternFilter}
                showOnlyFlaggedRings={showOnlyFlaggedRings}
                onToggleOnlyFlaggedRings={() => setShowOnlyFlaggedRings((v) => !v)}
              />

              {/* Compact Planted Fraud Rings Strip */}
              <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-3">
                {simulation.rings.map((ring) => {
                  const meanRisk = ring.meanRiskByModel[activeModelId];
                  const isRingSelected = ring.memberIds.includes(selectedAccount.id);
                  return (
                    <button
                      key={ring.ringId}
                      type="button"
                      onClick={() => setSelectedAccountId(ring.memberIds[0])}
                      className={`text-left p-3 rounded-lg border transition-colors ${
                        isRingSelected
                          ? 'bg-slate-800/90 border-amber-400/70'
                          : 'bg-slate-900 border-slate-800 hover:border-slate-700'
                      }`}
                    >
                      <div className="flex items-center justify-between text-xs font-mono tabular-nums">
                        <span className="font-semibold text-amber-300">{ring.ringId}</span>
                        <span className="text-slate-300">
                          Risk: {(meanRisk * 100).toFixed(0)}%
                        </span>
                      </div>
                      <div className="text-xs font-semibold text-slate-100 mt-1">
                        {ring.patternLabel}
                      </div>
                      <div className="text-[11px] text-slate-400 font-mono tabular-nums mt-1">
                        {ring.memberIds.length} accts · {ring.txCount} tx · $
                        {ring.totalVolume.toLocaleString()}
                      </div>
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Right 4 Columns: Per-Account Risk Card + SHAP + GNNExplainer */}
            <div className="lg:col-span-4">
              <AccountRiskInspector
                account={selectedAccount}
                features={selectedFeatures}
                activeEval={activeEval}
                allAccounts={simulation.accounts}
                onSelectAccount={setSelectedAccountId}
                shapAttributions={shapAttributions}
                gnnSubgraph={gnnSubgraph}
              />
            </div>
          </div>
        )}

        {activeTab === 'evaluation' && (
          <EvaluationAblationView
            evaluations={simulation.evaluations}
            activeModelId={activeModelId}
            onSelectModel={setActiveModelId}
            shiftMode={params.shiftMode}
            onToggleShiftMode={handleShiftToggle}
            leakageCheckPassed={simulation.leakageCheckPassed}
            trainCutoffHr={simulation.trainCutoffHr}
            valCutoffHr={simulation.valCutoffHr}
          />
        )}

        {activeTab === 'rings' && (
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-base font-semibold text-slate-100">
                  Planted Fraud Ring Topologies & Model Detection Matrix
                </h2>
                <p className="text-xs text-slate-400 mt-0.5">
                  Click any member account ID to inspect its SHAP attributions and 2-hop GNNExplainer subgraph in the Workbench.
                </p>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {simulation.rings.map((ring) => (
                <div
                  key={ring.ringId}
                  className="bg-slate-900 border border-slate-800 rounded-lg p-5 space-y-3"
                >
                  <div className="flex items-start justify-between gap-4">
                    <div>
                      <span className="text-xs font-mono font-semibold text-amber-300">
                        {ring.ringId}
                      </span>
                      <h3 className="text-base font-bold text-white mt-0.5">
                        {ring.patternLabel}
                      </h3>
                    </div>
                    <div className="text-right font-mono tabular-nums text-xs">
                      <div className="text-slate-200 font-semibold">
                        ${ring.totalVolume.toLocaleString()}
                      </div>
                      <div className="text-slate-400">{ring.txCount} planted edges</div>
                    </div>
                  </div>

                  <p className="text-xs text-slate-300 leading-relaxed">
                    {ring.signatureDescription}
                  </p>

                  <div className="pt-2 border-t border-slate-800">
                    <div className="text-xs text-slate-400 mb-1.5">
                      Ring Member Accounts (Click to Inspect):
                    </div>
                    <div className="flex flex-wrap gap-1.5">
                      {ring.memberIds.map((accId) => (
                        <button
                          key={accId}
                          type="button"
                          onClick={() => {
                            setSelectedAccountId(accId);
                            setActiveTab('workbench');
                          }}
                          className="px-2.5 py-1 text-xs font-mono bg-slate-950 hover:bg-slate-800 text-slate-200 border border-slate-700 rounded transition-colors"
                        >
                          {accId}
                        </button>
                      ))}
                    </div>
                  </div>

                  <div className="pt-2 border-t border-slate-800 grid grid-cols-4 gap-2 text-xs font-mono tabular-nums">
                    {(
                      [
                        ['xgboost_tabular', 'XGB Tab'],
                        ['xgboost_graph', 'XGB+Graph'],
                        ['graphsage_focal', 'GraphSAGE'],
                        ['gat_weighted', 'GAT'],
                      ] as const
                    ).map(([mId, label]) => {
                      const score = ring.meanRiskByModel[mId];
                      const caught = ring.detectedByModels.includes(mId);
                      return (
                        <div key={mId} className="bg-slate-950 p-2 rounded border border-slate-800">
                          <div className="text-slate-400 text-[11px]">{label}</div>
                          <div
                            className={`font-semibold mt-0.5 ${
                              caught ? 'text-emerald-400' : 'text-slate-400'
                            }`}
                          >
                            {(score * 100).toFixed(1)}%
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {activeTab === 'repository' && (
          <CodeAndReadmeExplorer simulation={simulation} />
        )}
      </main>
    </div>
  );
}
