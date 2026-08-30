import { useEffect, useMemo, useState } from "react";
import { Gauge, Timer, MemoryStick, Layers, ScanSearch, TrendingUp, ShieldAlert, ShieldCheck } from "lucide-react";
import { getMetrics, getHistory, type MetricsSnapshot, type HistoryEntry } from "@/lib/api/client";

interface DashboardKpisProps {
  refreshKey?: number;
}

interface Kpi {
  key: string;
  label: string;
  value: string;
  sub?: string;
  icon: React.ComponentType<{ className?: string }>;
  color: string;
}

/**
 * Top-of-dashboard KPI strip (Masterplan §5.3 + §5.5).
 *
 * Aggregates the live service snapshot (GET /api/v1/metrics) with the
 * audited history (GET /api/v1/history) to surface service health,
 * detection throughput, verdict mix, and per-image latency — all real
 * values from the FastAPI backend.
 */
const DashboardKpis = ({ refreshKey = 0 }: DashboardKpisProps) => {
  const [metrics, setMetrics] = useState<MetricsSnapshot | null>(null);
  const [history, setHistory] = useState<HistoryEntry[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        const [m, h] = await Promise.all([getMetrics(), getHistory(200)]);
        if (!cancelled) {
          setMetrics(m);
          setHistory(h);
          setError(null);
        }
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : "Live data unavailable");
      }
    };
    load();
    return () => {
      cancelled = true;
    };
  }, [refreshKey]);

  const kpis = useMemo<Kpi[]>(() => {
    const aiCount = history?.filter((h) => h.verdict === "ai").length ?? 0;
    const authenticCount = history?.filter((h) => h.verdict === "authentic").length ?? 0;
    const avgLatency =
      history && history.length
        ? history.reduce((sum, h) => sum + h.latency_ms, 0) / history.length
        : 0;

    return [
      {
        key: "throughput",
        label: "Throughput",
        value: metrics ? `${metrics.images_per_second.toFixed(2)} img/s` : "—",
        sub: "live inference",
        icon: Gauge,
        color: "text-sky-400",
      },
      {
        key: "p95",
        label: "p95 Latency",
        value: metrics ? `${metrics.latency_p95_ms.toFixed(0)} ms` : "—",
        sub: metrics ? `p50 ${metrics.latency_p50_ms.toFixed(0)} ms` : "live",
        icon: Timer,
        color: "text-cyan-400",
      },
      {
        key: "rss",
        label: "Peak RSS",
        value: metrics ? `${metrics.peak_rss_mb.toFixed(0)} MB` : "—",
        sub: "memory",
        icon: MemoryStick,
        color: "text-violet-400",
      },
      {
        key: "cache",
        label: "Cache",
        value: metrics ? `${metrics.cache_entries}` : "—",
        sub: "entries",
        icon: Layers,
        color: "text-amber-400",
      },
      {
        key: "analyzed",
        label: "Total Analyzed",
        value: metrics ? `${metrics.total_analyzed}` : history ? `${history.length}` : "—",
        sub: "images",
        icon: ScanSearch,
        color: "text-sky-400",
      },
      {
        key: "ai",
        label: "AI Detected",
        value: history ? `${aiCount}` : "—",
        sub: history ? `${((aiCount / Math.max(1, history.length)) * 100).toFixed(1)}% of samples` : "history",
        icon: ShieldAlert,
        color: "text-rose-400",
      },
      {
        key: "authentic",
        label: "Authentic",
        value: history ? `${authenticCount}` : "—",
        sub: history ? `${((authenticCount / Math.max(1, history.length)) * 100).toFixed(1)}% of samples` : "history",
        icon: ShieldCheck,
        color: "text-emerald-400",
      },
      {
        key: "avg-latency",
        label: "Mean Latency",
        value: history ? `${avgLatency.toFixed(0)} ms` : "—",
        sub: "per analysis",
        icon: TrendingUp,
        color: "text-cyan-400",
      },
    ];
  }, [metrics, history]);

  if (error) {
    return (
      <div className="rounded-xl border border-destructive/40 bg-destructive/5 p-4 text-sm text-destructive">
        {error} — is the FastAPI backend running on port 8000?
      </div>
    );
  }

  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-8">
      {kpis.map((kpi) => (
        <div
          key={kpi.key}
          className="glass rounded-xl p-4 transition-colors hover:border-primary/40 hover:bg-card/80"
        >
          <div className="flex items-center justify-between mb-2">
            <kpi.icon className={`h-5 w-5 ${kpi.color}`} />
            <span className="text-[10px] uppercase tracking-wider text-muted-foreground">
              {kpi.label}
            </span>
          </div>
          <div className="font-mono text-xl font-semibold leading-tight">{kpi.value}</div>
          <div className="text-[11px] text-muted-foreground mt-1">{kpi.sub}</div>
        </div>
      ))}
    </div>
  );
};

export default DashboardKpis;
