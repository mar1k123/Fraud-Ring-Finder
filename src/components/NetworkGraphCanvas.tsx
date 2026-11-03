import React, { useState, useMemo } from 'react';
import {
  ZoomIn,
  ZoomOut,
  Maximize2,
  Eye,
  Filter,
} from 'lucide-react';
import {
  AccountNode,
  TransactionEdge,
  FraudPattern,
  ModelEvaluationResult,
} from '../engine/fraudEngine';

interface NetworkGraphCanvasProps {
  accounts: AccountNode[];
  transactions: TransactionEdge[];
  activeEval: ModelEvaluationResult;
  selectedAccountId: string;
  onSelectAccount: (accountId: string) => void;
  patternFilter: FraudPattern | 'all';
  onChangePatternFilter: (pattern: FraudPattern | 'all') => void;
  showOnlyFlaggedRings: boolean;
  onToggleOnlyFlaggedRings: () => void;
}

const PATTERN_COLORS: Record<FraudPattern, string> = {
  legitimate: '#475569',
  money_mule_chain: '#EF4444',
  shared_device_ring: '#F59E0B',
  burst_activity: '#EC4899',
  circular_transfer: '#06B6D4',
};

const PATTERN_NAMES: Record<FraudPattern, string> = {
  legitimate: 'Legitimate',
  money_mule_chain: 'Money-Mule Chain',
  shared_device_ring: 'Shared-Device Ring',
  burst_activity: 'Burst Cluster',
  circular_transfer: 'Circular Loop',
};

export const NetworkGraphCanvas: React.FC<NetworkGraphCanvasProps> = ({
  accounts,
  transactions,
  activeEval,
  selectedAccountId,
  onSelectAccount,
  patternFilter,
  onChangePatternFilter,
  showOnlyFlaggedRings,
  onToggleOnlyFlaggedRings,
}) => {
  const [zoom, setZoom] = useState<number>(1.0);
  const [pan, setPan] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [hoveredId, setHoveredId] = useState<string | null>(null);

  const accountMap = useMemo(
    () => new Map(accounts.map((a) => [a.id, a])),
    [accounts]
  );

  const visibleAccounts = useMemo(() => {
    return accounts.filter((a) => {
      if (showOnlyFlaggedRings && !a.isFraud) return false;
      if (patternFilter !== 'all' && a.fraudPattern !== patternFilter) {
        return false;
      }
      return true;
    });
  }, [accounts, showOnlyFlaggedRings, patternFilter]);

  const visibleAccountSet = useMemo(
    () => new Set(visibleAccounts.map((a) => a.id)),
    [visibleAccounts]
  );

  const visibleEdges = useMemo(() => {
    // Keep all fraud edges + a clean representative sample of background edges so canvas stays crisp
    return transactions.filter((tx, idx) => {
      if (!visibleAccountSet.has(tx.srcId) || !visibleAccountSet.has(tx.dstId)) {
        return false;
      }
      if (tx.isFraud) return true;
      if (
        tx.srcId === selectedAccountId ||
        tx.dstId === selectedAccountId
      ) {
        return true;
      }
      return idx % 3 === 0;
    });
  }, [transactions, visibleAccountSet, selectedAccountId]);

  const handleResetView = () => {
    setZoom(1.0);
    setPan({ x: 0, y: 0 });
  };

  return (
    <div className="flex flex-col bg-slate-900 border border-slate-800 rounded-lg overflow-hidden">
      {/* Canvas Header & Filter Bar */}
      <div className="flex flex-wrap items-center justify-between gap-3 px-4 py-3 border-b border-slate-800 bg-slate-900/90">
        <div className="flex items-center gap-3">
          <h2 className="text-sm font-semibold text-slate-100">
            Interactive PyVis Transaction Topology
          </h2>
          <span className="text-xs text-slate-400 font-mono tabular-nums">
            {visibleAccounts.length} nodes · {visibleEdges.length} visible edges
          </span>
        </div>

        {/* Pattern Filter Controls */}
        <div className="flex flex-wrap items-center gap-1.5">
          <div className="flex items-center gap-1 p-1 bg-slate-950 border border-slate-800 rounded-md">
            {(
              [
                ['all', 'All Nodes'],
                ['money_mule_chain', 'Mule Chains'],
                ['shared_device_ring', 'Shared Device'],
                ['burst_activity', 'Burst Activity'],
                ['circular_transfer', 'Circular Loops'],
              ] as const
            ).map(([key, label]) => (
              <button
                key={key}
                type="button"
                onClick={() => onChangePatternFilter(key)}
                className={`px-2.5 py-1 text-xs font-medium rounded transition-colors whitespace-nowrap shrink-0 ${
                  patternFilter === key
                    ? 'bg-slate-800 text-amber-300'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                {label}
              </button>
            ))}
          </div>

          <button
            type="button"
            onClick={onToggleOnlyFlaggedRings}
            className={`flex items-center gap-1.5 px-2.5 py-1.5 text-xs font-medium rounded-md border transition-colors whitespace-nowrap shrink-0 ${
              showOnlyFlaggedRings
                ? 'bg-red-950/60 border-red-700/70 text-red-200'
                : 'bg-slate-950 border-slate-800 text-slate-300 hover:text-white'
            }`}
          >
            <Filter className="w-3.5 h-3.5" />
            {showOnlyFlaggedRings ? 'Showing Planted Rings Only' : 'Isolate Fraud Rings'}
          </button>

          <div className="flex items-center gap-1 bg-slate-950 border border-slate-800 rounded-md p-0.5">
            <button
              type="button"
              aria-label="Zoom in network graph"
              onClick={() => setZoom((z) => Math.min(2.2, Number((z + 0.2).toFixed(2))))}
              className="p-1.5 text-slate-400 hover:text-slate-100 transition-colors"
            >
              <ZoomIn className="w-3.5 h-3.5" />
            </button>
            <button
              type="button"
              aria-label="Zoom out network graph"
              onClick={() => setZoom((z) => Math.max(0.6, Number((z - 0.2).toFixed(2))))}
              className="p-1.5 text-slate-400 hover:text-slate-100 transition-colors"
            >
              <ZoomOut className="w-3.5 h-3.5" />
            </button>
            <button
              type="button"
              aria-label="Reset network graph viewport"
              onClick={handleResetView}
              className="p-1.5 text-slate-400 hover:text-slate-100 transition-colors"
            >
              <Maximize2 className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      </div>

      {/* Interactive SVG Topology Surface */}
      <div
        className="relative w-full h-[500px] bg-slate-950 select-none overflow-hidden"
        data-testid="pyvis-network-surface"
      >
        <svg
          viewBox="0 0 1000 700"
          className="w-full h-full"
          role="img"
          aria-label="Transaction network topology graph with highlighted fraud rings"
        >
          <defs>
            <marker
              id="arrow-legit"
              viewBox="0 0 10 10"
              refX="16"
              refY="5"
              markerWidth="4"
              markerHeight="4"
              orient="auto-start-reverse"
            >
              <path d="M 0 1 L 10 5 L 0 9 z" fill="#334155" />
            </marker>
            <marker
              id="arrow-fraud"
              viewBox="0 0 10 10"
              refX="18"
              refY="5"
              markerWidth="5"
              markerHeight="5"
              orient="auto-start-reverse"
            >
              <path d="M 0 1 L 10 5 L 0 9 z" fill="#EF4444" />
            </marker>
          </defs>

          {/* Subtle radial grid lines */}
          <g stroke="#1E293B" strokeWidth="1" opacity="0.55">
            <line x1="500" y1="0" x2="500" y2="700" strokeDasharray="4 4" />
            <line x1="0" y1="350" x2="1000" y2="350" strokeDasharray="4 4" />
            <circle cx="500" cy="350" r="180" fill="none" strokeDasharray="3 6" />
            <circle cx="500" cy="350" r="320" fill="none" strokeDasharray="3 6" />
          </g>

          <g
            transform={`translate(${500 + pan.x}, ${350 + pan.y}) scale(${zoom}) translate(-500, -350)`}
          >
            {/* Ring Zone Halos & Labels */}
            {patternFilter === 'all' && (
              <g>
                <rect
                  x="75"
                  y="120"
                  width="270"
                  height="165"
                  rx="12"
                  fill="#EF4444"
                  fillOpacity="0.04"
                  stroke="#EF4444"
                  strokeOpacity="0.3"
                  strokeDasharray="4 4"
                />
                <text
                  x="88"
                  y="140"
                  fill="#FCA5A5"
                  fontSize="11"
                  fontWeight="600"
                  fontFamily="IBM Plex Mono, monospace"
                >
                  RING-MULE-01 · Money-Mule Layering Chain
                </text>

                <circle
                  cx="780"
                  cy="210"
                  r="102"
                  fill="#F59E0B"
                  fillOpacity="0.04"
                  stroke="#F59E0B"
                  strokeOpacity="0.3"
                  strokeDasharray="4 4"
                />
                <text
                  x="690"
                  y="95"
                  fill="#FCD34D"
                  fontSize="11"
                  fontWeight="600"
                  fontFamily="IBM Plex Mono, monospace"
                >
                  RING-SDEV-02 · Shared Device (DEV-200)
                </text>

                <circle
                  cx="220"
                  cy="510"
                  r="102"
                  fill="#EC4899"
                  fillOpacity="0.04"
                  stroke="#EC4899"
                  strokeOpacity="0.3"
                  strokeDasharray="4 4"
                />
                <text
                  x="125"
                  y="632"
                  fill="#F9A8D4"
                  fontSize="11"
                  fontWeight="600"
                  fontFamily="IBM Plex Mono, monospace"
                >
                  RING-BURST-03 · High-Velocity Burst
                </text>

                <circle
                  cx="780"
                  cy="510"
                  r="102"
                  fill="#06B6D4"
                  fillOpacity="0.04"
                  stroke="#06B6D4"
                  strokeOpacity="0.3"
                  strokeDasharray="4 4"
                />
                <text
                  x="685"
                  y="632"
                  fill="#67E8F9"
                  fontSize="11"
                  fontWeight="600"
                  fontFamily="IBM Plex Mono, monospace"
                >
                  RING-CYCLE-04 · Circular Wash Loop
                </text>
              </g>
            )}

            {/* Directed Transaction Edges */}
            {visibleEdges.map((tx) => {
              const src = accountMap.get(tx.srcId);
              const dst = accountMap.get(tx.dstId);
              if (!src || !dst) return null;

              const isTouchingSelected =
                tx.srcId === selectedAccountId || tx.dstId === selectedAccountId;
              const strokeColor = tx.isFraud
                ? PATTERN_COLORS[tx.fraudPattern]
                : isTouchingSelected
                  ? '#94A3B8'
                  : '#1E293B';

              return (
                <line
                  key={tx.id}
                  x1={src.x}
                  y1={src.y}
                  x2={dst.x}
                  y2={dst.y}
                  stroke={strokeColor}
                  strokeWidth={tx.isFraud ? 2.1 : isTouchingSelected ? 1.5 : 0.9}
                  strokeOpacity={tx.isFraud ? 0.85 : isTouchingSelected ? 0.8 : 0.45}
                  markerEnd={tx.isFraud ? 'url(#arrow-fraud)' : 'url(#arrow-legit)'}
                />
              );
            })}

            {/* Account Nodes */}
            {visibleAccounts.map((acc) => {
              const score = activeEval.accountScores[acc.id] || 0;
              const isFlaggedByModel = score >= activeEval.thresholdAtTarget;
              const isSelected = acc.id === selectedAccountId;
              const isHovered = acc.id === hoveredId;
              const fillColor = PATTERN_COLORS[acc.fraudPattern];
              const radius = acc.isFraud ? 10 : isFlaggedByModel ? 8 : 5.5;

              return (
                <g
                  key={acc.id}
                  transform={`translate(${acc.x}, ${acc.y})`}
                  onClick={() => onSelectAccount(acc.id)}
                  onMouseEnter={() => setHoveredId(acc.id)}
                  onMouseLeave={() => setHoveredId(null)}
                  className="cursor-pointer"
                >
                  {/* Selection / Model Flag Halo */}
                  {(isSelected || isFlaggedByModel) && (
                    <circle
                      r={isSelected ? radius + 7 : radius + 4}
                      fill="none"
                      stroke={isSelected ? '#F8FAFC' : '#EF4444'}
                      strokeWidth={isSelected ? 2.2 : 1.3}
                      strokeDasharray={isSelected ? undefined : '2 2'}
                    />
                  )}

                  <circle
                    r={radius}
                    fill={fillColor}
                    stroke={isSelected ? '#FFFFFF' : '#0F172A'}
                    strokeWidth={isSelected ? 2 : 1.2}
                  />

                  {/* Always show labels for planted fraud ring nodes or selected/hovered node */}
                  {(acc.isFraud || isSelected || isHovered) && (
                    <text
                      y={-radius - 5}
                      textAnchor="middle"
                      fill={isSelected ? '#FFFFFF' : '#E2E8F0'}
                      fontSize={isSelected ? '11' : '9.5'}
                      fontWeight={isSelected || acc.isFraud ? '600' : '500'}
                      fontFamily="IBM Plex Mono, monospace"
                    >
                      {acc.id}
                    </text>
                  )}
                </g>
              );
            })}
          </g>
        </svg>

        {/* Bottom Overlay Legend & Quick Node Switcher */}
        <div className="absolute bottom-3 left-3 right-3 flex flex-wrap items-center justify-between gap-2 px-3.5 py-2 bg-slate-900/95 border border-slate-800 rounded-md text-xs text-slate-300">
          <div className="flex flex-wrap items-center gap-4">
            {(
              [
                ['money_mule_chain', '#EF4444', 'Money-Mule Chain'],
                ['shared_device_ring', '#F59E0B', 'Shared-Device Ring'],
                ['burst_activity', '#EC4899', 'Burst Cluster'],
                ['circular_transfer', '#06B6D4', 'Circular Loop'],
                ['legitimate', '#475569', 'Legitimate Account'],
              ] as const
            ).map(([key, color, label]) => (
              <span key={key} className="flex items-center gap-1.5">
                <span
                  className="w-2.5 h-2.5 rounded-sm shrink-0"
                  style={{ backgroundColor: color }}
                />
                <span>{label}</span>
              </span>
            ))}
          </div>

          <div className="flex items-center gap-2 text-slate-400 font-mono tabular-nums">
            <Eye className="w-3.5 h-3.5 text-amber-400" />
            <span>
              Selected: <strong className="text-slate-100">{selectedAccountId}</strong> (
              {PATTERN_NAMES[accountMap.get(selectedAccountId)?.fraudPattern || 'legitimate']})
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};
