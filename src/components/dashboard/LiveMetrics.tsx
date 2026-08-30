import { useEffect, useState } from "react";
import type { MetricsSnapshot } from "@/lib/api/client";
import { getMetrics } from "@/lib/api/client";
import { Activity, Gauge, Timer, MemoryStick, Layers } from "lucide-react";

interface LiveMetricsProps {
  /** Poll interval in ms. Defaults to 3000ms. */
  intervalMs?: number;
  /** Optional ref callback so the parent can force a refresh after an analysis. */
  refreshKey?: number;
}

/**
 * Real-time metrics panel (Masterplan §5.3) — streams live service snapshot
 * from GET /api/v1/metrics. Shows throughput, p50/p95/p99 latency, peak RSS,
 * and cache entries.
 */
const LiveMetrics = ({ intervalMs = 3000, refreshKey = 0 }: LiveMetricsProps) => {
  const [metrics, setMetrics] = useState<MetricsSnapshot | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        const snap = await getMetrics();
        if (!cancelled) {
          setMetrics(snap);
          setError(null);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Metrics unavailable");
        }
      }
    };
    load();
    const timer = setInterval(load, intervalMs);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [intervalMs, refreshKey]);

  return (
    <div className="glass rounded-xl p-6">
      <div className="mb-4 flex items-center gap-2">
        <Activity className="h-5 w-5 text-primary" />
        <h3 className="text-lg font-semibold">Live Service Metrics</h3>
        <span className="ml-auto text-xs text-muted-foreground flex items-center gap-1">
          <span className="h-2 w-2 rounded-full bg-success animate-pulse" />
          polling {intervalMs / 1000}s
        </span>
      </div>

      {error && <p className="text-sm text-destructive">{error} — is the API running?</p>}

      {metrics ? (
        <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
          <MetricTile icon={Gauge} label="Throughput" value={`${metrics.images_per_second.toFixed(2)} img/s`} />
          <MetricTile icon={Timer} label="p50" value={`${metrics.latency_p50_ms.toFixed(0)} ms`} />
          <MetricTile icon={Timer} label="p95" value={`${metrics.latency_p95_ms.toFixed(0)} ms`} />
          <MetricTile icon={MemoryStick} label="Peak RSS" value={`${metrics.peak_rss_mb.toFixed(0)} MB`} />
          <MetricTile icon={Layers} label="Cache" value={`${metrics.cache_entries} entries`} />
        </div>
      ) : (
        !error && <p className="text-sm text-muted-foreground">Loading metrics…</p>
      )}

      <div className="mt-4 flex justify-between text-xs text-muted-foreground">
        <span>Analyzed: {metrics?.total_analyzed ?? 0}</span>
        <span>Uptime: {(metrics?.uptime_seconds ?? 0).toFixed(0)}s</span>
      </div>
    </div>
  );
};

function MetricTile({
  icon: Icon,
  label,
  value,
}: {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  value: string;
}) {
  return (
    <div className="rounded-lg bg-muted/30 p-3">
      <Icon className="h-4 w-4 text-muted-foreground mb-2" />
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className="font-mono text-lg font-medium">{value}</div>
    </div>
  );
}

export default LiveMetrics;
