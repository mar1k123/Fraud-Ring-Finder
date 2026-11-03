import React, { useState } from 'react';
import { Copy, Check, FileCode, BookOpen, Terminal, CheckCircle2 } from 'lucide-react';
import { REPO_FILES } from '../data/pythonSourceFiles';
import { SimulationBundle } from '../engine/fraudEngine';

interface CodeAndReadmeExplorerProps {
  simulation: SimulationBundle;
  initialSubTab?: 'readme' | 'code';
}

export const CodeAndReadmeExplorer: React.FC<CodeAndReadmeExplorerProps> = ({
  simulation,
  initialSubTab = 'readme',
}) => {
  const [subTab, setSubTab] = useState<'readme' | 'code'>(initialSubTab);
  const [selectedFilePath, setSelectedFilePath] = useState<string>(REPO_FILES[1].path);
  const [copied, setCopied] = useState<boolean>(false);

  const activeFile =
    REPO_FILES.find((f) => f.path === selectedFilePath) || REPO_FILES[0];

  const handleCopy = () => {
    navigator.clipboard?.writeText(activeFile.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 1800);
  };

  const tabEv = simulation.evaluations.xgboost_tabular;
  const graphEv = simulation.evaluations.xgboost_graph;
  const sageEv = simulation.evaluations.graphsage_focal;
  const gatEv = simulation.evaluations.gat_weighted;

  return (
    <div className="space-y-4">
      {/* Sub-navigation between Architecture README and Full Python Source Browser */}
      <div className="flex items-center justify-between gap-4 bg-slate-900 border border-slate-800 rounded-lg px-4 py-2.5">
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => setSubTab('readme')}
            className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-md transition-colors ${
              subTab === 'readme'
                ? 'bg-slate-800 text-white'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <BookOpen className="w-3.5 h-3.5" />
            README & Real Data Integration Guide
          </button>
          <button
            type="button"
            onClick={() => setSubTab('code')}
            className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-md transition-colors ${
              subTab === 'code'
                ? 'bg-slate-800 text-white'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <FileCode className="w-3.5 h-3.5" />
            Python / PyTorch Geometric Source Files ({REPO_FILES.length})
          </button>
        </div>

        <div className="text-xs text-slate-400 font-mono tabular-nums">
          All numbers below are live-computed from seed={simulation.params.seed}
        </div>
      </div>

      {subTab === 'readme' ? (
        <div className="bg-slate-900 border border-slate-800 rounded-lg p-6 space-y-6 text-slate-200 text-sm leading-relaxed">
          <div className="border-b border-slate-800 pb-4">
            <h2 className="text-2xl font-bold text-white">
              fraud-ring-finder — Graph ML Transaction Forensics
            </h2>
            <p className="text-slate-400 mt-1">
              End-to-end Graph Machine Learning benchmark and interactive workbench for detecting collusive fraud rings in financial transaction networks using PyTorch Geometric (GraphSAGE, GAT), XGBoost, SHAP, and GNNExplainer.
            </p>
          </div>

          {/* Section 1: Why Synthetic Data & What That Implies */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="bg-slate-950 border border-slate-800 rounded-lg p-4">
              <h3 className="text-base font-semibold text-white mb-2">
                1. Why Synthetic Data & What That Implies
              </h3>
              <p className="text-xs text-slate-300 leading-relaxed mb-2">
                Real-world payment graphs containing confirmed money-mule chains, device fingerprints, and wire transfers are strictly protected under banking secrecy and PII regulations (GLBA, GDPR, PCI-DSS). Public datasets either strip graph topology into anonymized PCA columns or lack ground-truth ring identifiers.
              </p>
              <p className="text-xs text-slate-400 leading-relaxed">
                <strong className="text-slate-200">What this implies:</strong> Because fraud patterns here are constructively planted via <code className="text-amber-300">src/generator/synthetic_network.py</code>, we have exact ground-truth ring membership for ablation studies and distribution-shift stress tests. However, synthetic background noise is cleaner than production banking traffic, so real deployments should expect higher false-positive pressure from legitimate corporate treasury sweeps and payroll batches.
              </p>
            </div>

            <div className="bg-slate-950 border border-slate-800 rounded-lg p-4">
              <h3 className="text-base font-semibold text-white mb-2">
                2. How to Plug In Real Transaction Data
              </h3>
              <p className="text-xs text-slate-300 leading-relaxed mb-2">
                To run the pipeline on your own ledger data, provide two Parquet/CSV tables matching the schema in <code className="text-amber-300">src/features/graph_features.py</code>:
              </p>
              <ul className="text-xs text-slate-300 space-y-1 font-mono">
                <li>
                  • accounts.parquet: account_id, account_age_days, kyc_tier, declared_monthly_vol
                </li>
                <li>
                  • transactions.parquet: tx_id, src_id, dst_id, device_id, merchant_id, timestamp_hr, amount, is_fraud
                </li>
              </ul>
              <p className="text-xs text-slate-400 mt-2">
                Pass both DataFrames into <code className="text-cyan-300">temporal_train_val_test_split()</code> followed by <code className="text-cyan-300">build_features_up_to_time()</code> to guarantee zero future data leakage.
              </p>
            </div>
          </div>

          {/* Section 2: Script-Generated Live Results Table */}
          <div>
            <h3 className="text-base font-semibold text-white mb-2">
              3. Script-Generated Benchmark Results (Seed {simulation.params.seed}, Fraud Rate {(simulation.params.fraudRate * 100).toFixed(2)}%)
            </h3>
            <div className="overflow-x-auto border border-slate-800 rounded-lg">
              <table className="w-full text-left border-collapse text-xs font-mono tabular-nums">
                <thead>
                  <tr className="bg-slate-950 border-b border-slate-800 text-slate-400">
                    <th className="py-2.5 px-3">Model</th>
                    <th className="py-2.5 px-3 text-right">PR-AUC</th>
                    <th className="py-2.5 px-3 text-right">Recall@80%Prec</th>
                    <th className="py-2.5 px-3 text-right">Mule Chain</th>
                    <th className="py-2.5 px-3 text-right">Shared Device</th>
                    <th className="py-2.5 px-3 text-right">Burst Cluster</th>
                    <th className="py-2.5 px-3 text-right">Circular Loop</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800 bg-slate-950/40">
                  {[tabEv, graphEv, sageEv, gatEv].map((ev) => (
                    <tr key={ev.modelId}>
                      <td className="py-2 px-3 font-sans font-medium text-slate-200">
                        {ev.modelName}
                      </td>
                      <td className="py-2 px-3 text-right text-white font-semibold">
                        {ev.prAuc.toFixed(4)}
                      </td>
                      <td className="py-2 px-3 text-right text-emerald-400">
                        {(ev.recallAtFixedPrecision * 100).toFixed(1)}%
                      </td>
                      <td className="py-2 px-3 text-right">
                        {(ev.patternRecall.money_mule_chain * 100).toFixed(0)}%
                      </td>
                      <td className="py-2 px-3 text-right">
                        {(ev.patternRecall.shared_device_ring * 100).toFixed(0)}%
                      </td>
                      <td className="py-2 px-3 text-right">
                        {(ev.patternRecall.burst_activity * 100).toFixed(0)}%
                      </td>
                      <td className="py-2 px-3 text-right">
                        {(ev.patternRecall.circular_transfer * 100).toFixed(0)}%
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Section 3: Verification & Unit Test Checklist */}
          <div className="bg-slate-950 border border-slate-800 rounded-lg p-4">
            <h3 className="text-sm font-semibold text-white mb-2 flex items-center gap-2">
              <Terminal className="w-4 h-4 text-emerald-400" />
              Automated Determinism & Zero-Leakage Verification Suite
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-2 text-xs text-slate-300">
              <div className="flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
                <span>
                  <code className="text-slate-100">test_generator_is_strictly_deterministic</code>: Identical seeds yield byte-identical graphs.
                </span>
              </div>
              <div className="flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
                <span>
                  <code className="text-slate-100">test_no_future_leakage_in_feature_extraction</code>: Mutating test edges has zero effect on train features.
                </span>
              </div>
            </div>
          </div>
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 items-stretch">
          {/* File Tree Sidebar */}
          <div className="lg:col-span-4 bg-slate-900 border border-slate-800 rounded-lg p-3 space-y-1">
            <div className="text-xs font-semibold text-slate-400 px-2 py-1.5">
              Project Repository Files
            </div>
            {REPO_FILES.map((file) => {
              const active = file.path === selectedFilePath;
              return (
                <button
                  key={file.path}
                  type="button"
                  onClick={() => setSelectedFilePath(file.path)}
                  className={`w-full text-left px-3 py-2 rounded-md text-xs transition-colors flex items-center justify-between ${
                    active
                      ? 'bg-slate-800 text-amber-300 font-medium'
                      : 'text-slate-300 hover:bg-slate-800/50'
                  }`}
                >
                  <span className="font-mono truncate">{file.path}</span>
                  <span className="text-slate-500 text-[11px] ml-2 shrink-0">
                    {file.category}
                  </span>
                </button>
              );
            })}
          </div>

          {/* Code Viewer */}
          <div className="lg:col-span-8 bg-slate-900 border border-slate-800 rounded-lg overflow-hidden flex flex-col">
            <div className="flex items-center justify-between px-4 py-3 border-b border-slate-800 bg-slate-950">
              <div>
                <div className="text-xs font-mono font-semibold text-amber-300">
                  /{activeFile.path}
                </div>
                <div className="text-xs text-slate-400 mt-0.5">{activeFile.description}</div>
              </div>
              <button
                type="button"
                onClick={handleCopy}
                className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium bg-slate-800 hover:bg-slate-700 text-slate-100 rounded-md transition-colors shrink-0"
              >
                {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                {copied ? 'Copied' : 'Copy File'}
              </button>
            </div>
            <pre className="p-4 text-xs font-mono text-slate-200 bg-slate-950/90 overflow-x-auto leading-relaxed max-h-[540px]">
              <code>{activeFile.content}</code>
            </pre>
          </div>
        </div>
      )}
    </div>
  );
};
