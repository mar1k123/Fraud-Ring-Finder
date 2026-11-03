import React from 'react';
import { ShieldAlert, ShieldCheck, Network, GitCommit } from 'lucide-react';
import {
  AccountNode,
  AccountFeatures,
  ModelEvaluationResult,
  ShapFeatureAttribution,
  GnnSubgraphEdge,
} from '../engine/fraudEngine';

interface AccountRiskInspectorProps {
  account: AccountNode;
  features: AccountFeatures;
  activeEval: ModelEvaluationResult;
  allAccounts: AccountNode[];
  onSelectAccount: (accountId: string) => void;
  shapAttributions: ShapFeatureAttribution[];
  gnnSubgraph: GnnSubgraphEdge[];
}

export const AccountRiskInspector: React.FC<AccountRiskInspectorProps> = ({
  account,
  features,
  activeEval,
  allAccounts,
  onSelectAccount,
  shapAttributions,
  gnnSubgraph,
}) => {
  const riskScore = activeEval.accountScores[account.id] ?? 0;
  const isFlagged = riskScore >= activeEval.thresholdAtTarget;

  // Quick selector sorted with fraud ring nodes first so analysts can inspect any account
  const sortedAccounts = [...allAccounts].sort((a, b) => {
    const scoreB = activeEval.accountScores[b.id] || 0;
    const scoreA = activeEval.accountScores[a.id] || 0;
    return scoreB - scoreA;
  });

  return (
    <div className="flex flex-col gap-4 bg-slate-900 border border-slate-800 rounded-lg p-4">
      {/* Header & Account Selector */}
      <div className="flex flex-col gap-2.5 pb-3 border-b border-slate-800">
        <div className="flex items-center justify-between gap-2">
          <span className="text-xs font-medium text-slate-400">
            Per-Account Risk Card & Explainability
          </span>
          <label className="sr-only" htmlFor="inspector-account-select">
            Inspect Account ID
          </label>
          <select
            id="inspector-account-select"
            aria-label="Inspect Account ID"
            value={account.id}
            onChange={(e) => onSelectAccount(e.target.value)}
            className="bg-slate-950 border border-slate-700 text-slate-100 text-xs font-mono rounded px-2.5 py-1.5 focus:outline-none focus:border-amber-400"
          >
            {sortedAccounts.map((acc) => {
              const s = activeEval.accountScores[acc.id] || 0;
              return (
                <option key={acc.id} value={acc.id}>
                  {acc.id} · {(s * 100).toFixed(1)}% ·{' '}
                  {acc.ringId !== 'NONE' ? acc.ringId : 'LEGIT'}
                </option>
              );
            })}
          </select>
        </div>

        {/* Account Risk Headline */}
        <div className="flex items-baseline justify-between gap-4 pt-1">
          <div>
            <div className="flex items-center gap-2">
              <h3 className="text-xl font-bold text-slate-100 font-mono tabular-nums">
                {account.id}
              </h3>
              {isFlagged ? (
                <span className="inline-flex items-center gap-1 text-xs font-semibold text-red-400">
                  <ShieldAlert className="w-4 h-4" />
                  Flagged Above Threshold
                </span>
              ) : (
                <span className="inline-flex items-center gap-1 text-xs font-medium text-emerald-400">
                  <ShieldCheck className="w-4 h-4" />
                  Below Alert Threshold
                </span>
              )}
            </div>
            <p className="text-xs text-slate-400 mt-1">
              Ring: <span className="font-mono text-slate-200">{account.ringId}</span> · Pattern:{' '}
              <span className="font-mono text-amber-300">{account.fraudPattern}</span> · Device:{' '}
              <span className="font-mono text-slate-200">{account.primaryDeviceId}</span>
            </p>
          </div>

          <div className="text-right">
            <div
              className={`text-2xl font-bold font-mono tabular-nums ${
                isFlagged ? 'text-red-400' : 'text-emerald-400'
              }`}
              data-testid="selected-account-risk-score"
            >
              {(riskScore * 100).toFixed(1)}%
            </div>
            <div className="text-xs text-slate-400 font-mono tabular-nums">
              Cutoff: {(activeEval.thresholdAtTarget * 100).toFixed(1)}%
            </div>
          </div>
        </div>
      </div>

      {/* Key Topology & Velocity Telemetry Row */}
      <div className="grid grid-cols-4 gap-2 text-xs pb-3 border-b border-slate-800">
        <div>
          <div className="text-slate-400">Shared Dev</div>
          <div className="font-mono font-semibold text-slate-100 tabular-nums mt-0.5">
            {features.sharedDeviceAccounts} accts
          </div>
        </div>
        <div>
          <div className="text-slate-400">1h Burst</div>
          <div className="font-mono font-semibold text-slate-100 tabular-nums mt-0.5">
            {features.burstTx1hMax} tx/hr
          </div>
        </div>
        <div>
          <div className="text-slate-400">PageRank</div>
          <div className="font-mono font-semibold text-slate-100 tabular-nums mt-0.5">
            {features.pageRank.toFixed(4)}
          </div>
        </div>
        <div>
          <div className="text-slate-400">Cycles (L≤4)</div>
          <div className="font-mono font-semibold text-slate-100 tabular-nums mt-0.5">
            {features.cycleParticipation}
          </div>
        </div>
      </div>

      {/* Section 1: SHAP Feature Attributions (XGBoost TreeExplainer) */}
      <div>
        <div className="flex items-center justify-between mb-2">
          <h4 className="text-xs font-semibold text-slate-200 flex items-center gap-1.5">
            <GitCommit className="w-3.5 h-3.5 text-amber-400" />
            SHAP Feature Attributions (Log-Odds Impact)
          </h4>
          <span className="text-xs text-slate-400">TreeExplainer</span>
        </div>

        <div className="space-y-1.5">
          {shapAttributions.slice(0, 6).map((item) => {
            const isPositive = item.shapDelta >= 0;
            const barWidthPct = Math.min(100, Math.round(Math.abs(item.shapDelta) * 55));

            return (
              <div key={item.feature} className="text-xs">
                <div className="flex items-center justify-between text-slate-300 mb-0.5">
                  <span className="truncate">
                    {item.label}{' '}
                    <span className="text-slate-500 font-mono">({item.formattedValue})</span>
                  </span>
                  <span
                    className={`font-mono tabular-nums font-medium ${
                      isPositive ? 'text-red-400' : 'text-sky-400'
                    }`}
                  >
                    {isPositive ? `+${item.shapDelta.toFixed(3)}` : item.shapDelta.toFixed(3)}
                  </span>
                </div>
                <div className="w-full h-1.5 bg-slate-950 rounded overflow-hidden">
                  <div
                    className={`h-full rounded ${
                      isPositive ? 'bg-red-500/80' : 'bg-sky-500/75'
                    }`}
                    style={{ width: `${ Math.max(6, barWidthPct) }%` }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Section 2: GNNExplainer 2-Hop Minimal Subgraph Mask */}
      <div className="pt-3 border-t border-slate-800">
        <div className="flex items-center justify-between mb-2">
          <h4 className="text-xs font-semibold text-slate-200 flex items-center gap-1.5">
            <Network className="w-3.5 h-3.5 text-cyan-400" />
            GNNExplainer Top Subgraph Edges
          </h4>
          <span className="text-xs text-slate-400">2-Hop Edge Mask</span>
        </div>

        <div className="divide-y divide-slate-800/80 border border-slate-800 rounded bg-slate-950/60">
          {gnnSubgraph.slice(0, 5).map((edge) => (
            <div key={edge.edgeId} className="px-2.5 py-2 text-xs flex flex-col gap-0.5">
              <div className="flex items-center justify-between font-mono tabular-nums">
                <span className="text-slate-200 font-medium">
                  {edge.srcId} → {edge.dstId}
                </span>
                <span className="text-amber-300 font-semibold">
                  mask={edge.importanceWeight.toFixed(2)} · ${edge.amount.toLocaleString()}
                </span>
              </div>
              <div className="text-slate-400 flex items-center justify-between">
                <span>{edge.reason}</span>
                <span className="font-mono tabular-nums text-slate-500">
                  t={edge.timestampHr.toFixed(1)}h
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
