import React from 'react';
import {
  ModelId,
  ModelEvaluationResult,
  ShiftMode,
} from '../engine/fraudEngine';

interface EvaluationAblationViewProps {
  evaluations: Record<ModelId, ModelEvaluationResult>;
  activeModelId: ModelId;
  onSelectModel: (modelId: ModelId) => void;
  shiftMode: ShiftMode;
  onToggleShiftMode: (mode: ShiftMode) => void;
  leakageCheckPassed: boolean;
  trainCutoffHr: number;
  valCutoffHr: number;
}

const MODEL_ORDER: ModelId[] = [
  'xgboost_tabular',
  'xgboost_graph',
  'graphsage_focal',
  'gat_weighted',
];

const MODEL_COLORS: Record<ModelId, string> = {
  xgboost_tabular: '#94A3B8',
  xgboost_graph: '#F59E0B',
  graphsage_focal: '#10B981',
  gat_weighted: '#38BDF8',
};

export const EvaluationAblationView: React.FC<EvaluationAblationViewProps> = ({
  evaluations,
  activeModelId,
  onSelectModel,
  shiftMode,
  onToggleShiftMode,
  leakageCheckPassed,
  trainCutoffHr,
  valCutoffHr,
}) => {
  return (
    <div className="space-y-6">
      {/* Top Banner: Temporal Split & Distribution Shift Controls */}
      <div className="flex flex-wrap items-center justify-between gap-4 p-4 bg-slate-900 border border-slate-800 rounded-lg">
        <div>
          <h2 className="text-base font-semibold text-slate-100">
            Temporal Split Protocol & Distribution Shift Robustness
          </h2>
          <p className="text-xs text-slate-400 mt-0.5 font-mono tabular-nums">
            Train Window: 0.0h – {trainCutoffHr.toFixed(1)}h (60%) · Validation:{' '}
            {trainCutoffHr.toFixed(1)}h – {valCutoffHr.toFixed(1)}h (20%) · Test:{' '}
            {valCutoffHr.toFixed(1)}h – 168.0h (20%) · Leakage Invariance:{' '}
            <span className="text-emerald-400 font-semibold">
              {leakageCheckPassed ? 'VERIFIED (0 Future Edges Leaked)' : 'FAILED'}
            </span>
          </p>
        </div>

        <div className="flex items-center gap-2 bg-slate-950 p-1 border border-slate-800 rounded-lg">
          <button
            type="button"
            onClick={() => onToggleShiftMode('none')}
            className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap shrink-0 ${
              shiftMode === 'none'
                ? 'bg-slate-800 text-white'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Stationary Test Distribution
          </button>
          <button
            type="button"
            onClick={() => onToggleShiftMode('tactic_shift')}
            className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap shrink-0 ${
              shiftMode === 'tactic_shift'
                ? 'bg-amber-500/20 text-amber-300 border border-amber-500/40'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Post-Training Tactic Shift (Device Rotation + Throttled Burst)
          </button>
        </div>
      </div>

      {/* Main Model Comparison Table */}
      <div className="bg-slate-900 border border-slate-800 rounded-lg overflow-hidden">
        <div className="px-4 py-3 border-b border-slate-800 flex items-center justify-between">
          <h3 className="text-sm font-semibold text-slate-100">
            Model Benchmark: Baselines vs PyTorch Geometric GNNs
          </h3>
          <span className="text-xs text-slate-400 font-mono tabular-nums">
            Click any model row to set active scoring model
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-xs">
            <thead>
              <tr className="border-b border-slate-800 text-slate-400 bg-slate-950/60">
                <th className="py-2.5 px-4 font-medium">Architecture</th>
                <th className="py-2.5 px-4 font-medium">Imbalance Loss Strategy</th>
                <th className="py-2.5 px-4 font-medium text-right">PR-AUC</th>
                <th className="py-2.5 px-4 font-medium text-right">
                  Recall @ {(evaluations.xgboost_tabular.fixedPrecisionTarget * 100).toFixed(0)}% Precision
                </th>
                <th className="py-2.5 px-4 font-medium text-right">Pre-Shift PR-AUC</th>
                <th className="py-2.5 px-4 font-medium text-right">Post-Shift PR-AUC</th>
                <th className="py-2.5 px-4 font-medium text-right">Shift Degradation</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/80 font-mono tabular-nums">
              {MODEL_ORDER.map((mId) => {
                const ev = evaluations[mId];
                const isSelected = mId === activeModelId;
                return (
                  <tr
                    key={mId}
                    onClick={() => onSelectModel(mId)}
                    className={`cursor-pointer transition-colors ${
                      isSelected ? 'bg-slate-800/70' : 'hover:bg-slate-800/30'
                    }`}
                  >
                    <td className="py-3 px-4 font-sans font-semibold text-slate-100">
                      {ev.modelName}
                      {isSelected && (
                        <span className="ml-2 text-xs font-mono text-amber-300">
                          [Active]
                        </span>
                      )}
                    </td>
                    <td className="py-3 px-4 font-sans text-slate-300">{ev.lossFunction}</td>
                    <td className="py-3 px-4 text-right font-semibold text-slate-100">
                      {ev.prAuc.toFixed(4)}
                    </td>
                    <td className="py-3 px-4 text-right font-semibold text-emerald-400">
                      {(ev.recallAtFixedPrecision * 100).toFixed(1)}%
                    </td>
                    <td className="py-3 px-4 text-right text-slate-300">
                      {ev.preShiftPrAuc.toFixed(4)}
                    </td>
                    <td className="py-3 px-4 text-right text-amber-300">
                      {ev.postShiftPrAuc.toFixed(4)}
                    </td>
                    <td className="py-3 px-4 text-right text-red-400">
                      -{ev.shiftDropPct.toFixed(1)}%
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* 2-Column Split: Per-Pattern Ablation Matrix + Precision-Recall Curve */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 items-stretch">
        {/* Per-Pattern Detection Ablation Table */}
        <div className="bg-slate-900 border border-slate-800 rounded-lg p-4 flex flex-col justify-between">
          <div>
            <h3 className="text-sm font-semibold text-slate-100">
              Ablation Study: Recall by Planted Fraud Pattern
            </h3>
            <p className="text-xs text-slate-400 mt-1 mb-3">
              Demonstrates which structural fraud topologies require hand-crafted graph features vs multi-hop GNN message passing.
            </p>

            <table className="w-full text-left border-collapse text-xs">
              <thead>
                <tr className="border-b border-slate-800 text-slate-400">
                  <th className="py-2 pr-2 font-medium">Planted Pattern</th>
                  <th className="py-2 px-2 font-medium text-right">XGB Tabular</th>
                  <th className="py-2 px-2 font-medium text-right">XGB + Graph</th>
                  <th className="py-2 px-2 font-medium text-right">GraphSAGE</th>
                  <th className="py-2 pl-2 font-medium text-right">GAT</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/70 font-mono tabular-nums">
                {(
                  [
                    ['burst_activity', '1. High-Velocity Burst'],
                    ['shared_device_ring', '2. Shared-Device Ring'],
                    ['circular_transfer', '3. Circular Wash Loop'],
                    ['money_mule_chain', '4. Money-Mule 5-Hop Chain'],
                  ] as const
                ).map(([patKey, patLabel]) => (
                  <tr key={patKey}>
                    <td className="py-2.5 pr-2 font-sans font-medium text-slate-200">
                      {patLabel}
                    </td>
                    {MODEL_ORDER.map((mId) => {
                      const rec = evaluations[mId].patternRecall[patKey] || 0;
                      const colorClass =
                        rec >= 0.75
                          ? 'text-emerald-400 font-semibold'
                          : rec >= 0.45
                            ? 'text-amber-300'
                            : 'text-slate-500';
                      return (
                        <td key={mId} className={`py-2.5 px-2 text-right ${colorClass}`}>
                          {(rec * 100).toFixed(0)}%
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="mt-4 pt-3 border-t border-slate-800 text-xs text-slate-400 leading-relaxed">
            <strong className="text-slate-200">Key Takeaway:</strong> Tabular XGBoost catches{' '}
            <span className="text-slate-200">Burst Activity</span> via 1-hour velocity features,
            but misses camouflaged <span className="text-slate-200">Money-Mule Chains</span> and{' '}
            <span className="text-slate-200">Circular Transfers</span>. Adding PageRank and
            shared-device counts recovers device rings, while 2-layer GraphSAGE and GAT capture
            multi-hop mule layering chains even after post-training device rotation.
          </div>
        </div>

        {/* Precision-Recall Curve SVG */}
        <div className="bg-slate-900 border border-slate-800 rounded-lg p-4 flex flex-col justify-between">
          <div className="flex items-center justify-between mb-2">
            <div>
              <h3 className="text-sm font-semibold text-slate-100">
                Precision–Recall Curve (No-Leakage Temporal Test Set)
              </h3>
              <p className="text-xs text-slate-400">
                Dashed horizontal line marks fixed precision target (
                {(evaluations.xgboost_tabular.fixedPrecisionTarget * 100).toFixed(0)}%)
              </p>
            </div>
          </div>

          <svg
            viewBox="0 0 460 240"
            className="w-full h-56 bg-slate-950 border border-slate-800 rounded p-2"
            role="img"
            aria-label="Precision Recall curves comparing XGBoost, GraphSAGE, and GAT"
          >
            {/* Axes */}
            <line x1="45" y1="20" x2="45" y2="205" stroke="#334155" strokeWidth="1" />
            <line x1="45" y1="205" x2="435" y2="205" stroke="#334155" strokeWidth="1" />

            {/* Fixed Precision Target Line */}
            {(() => {
              const targetY =
                205 - evaluations.xgboost_tabular.fixedPrecisionTarget * 180;
              return (
                <g>
                  <line
                    x1="45"
                    y1={targetY}
                    x2="435"
                    y2={targetY}
                    stroke="#F59E0B"
                    strokeWidth="1"
                    strokeDasharray="4 4"
                  />
                  <text
                    x="430"
                    y={targetY - 5}
                    textAnchor="end"
                    fill="#FCD34D"
                    fontSize="9"
                    fontFamily="IBM Plex Mono, monospace"
                  >
                    Target Precision {(evaluations.xgboost_tabular.fixedPrecisionTarget * 100).toFixed(0)}%
                  </text>
                </g>
              );
            })()}

            {/* Plot Curves */}
            {MODEL_ORDER.map((mId) => {
              const pts = evaluations[mId].prCurve;
              const d = pts
                .map((p, i) => {
                  const x = 45 + p.recall * 385;
                  const y = 205 - p.precision * 180;
                  return `${i === 0 ? 'M' : 'L'} ${x.toFixed(1)} ${y.toFixed(1)}`;
                })
                .join(' ');
              return (
                <path
                  key={mId}
                  d={d}
                  fill="none"
                  stroke={MODEL_COLORS[mId]}
                  strokeWidth={mId === activeModelId ? 2.6 : 1.5}
                  opacity={mId === activeModelId ? 1 : 0.75}
                />
              );
            })}

            <text
              x="240"
              y="228"
              textAnchor="middle"
              fill="#94A3B8"
              fontSize="10"
              fontFamily="IBM Plex Mono, monospace"
            >
              Recall (Fraction of Fraud Ring Accounts Caught)
            </text>
            <text
              x="14"
              y="115"
              textAnchor="middle"
              fill="#94A3B8"
              fontSize="10"
              fontFamily="IBM Plex Mono, monospace"
              transform="rotate(-90 14 115)"
            >
              Precision
            </text>
          </svg>

          <div className="flex flex-wrap items-center justify-between gap-3 mt-3 text-xs font-mono">
            {(
              [
                ['xgboost_tabular', 'XGB Tabular'],
                ['xgboost_graph', 'XGB + Graph'],
                ['graphsage_focal', 'GraphSAGE'],
                ['gat_weighted', 'GAT'],
              ] as const
            ).map(([mId, shortLabel]) => (
              <span key={mId} className="flex items-center gap-1.5 text-slate-300">
                <span
                  className="w-3 h-1 rounded-sm"
                  style={{ backgroundColor: MODEL_COLORS[mId] }}
                />
                {shortLabel} ({evaluations[mId].prAuc.toFixed(3)})
              </span>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};
