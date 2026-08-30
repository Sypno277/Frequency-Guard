import { useEffect, useMemo, useState } from "react";
import {
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
  Tooltip,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  LineChart,
  Line,
  Legend,
} from "recharts";
import { getHistory, type HistoryEntry } from "@/lib/api/client";
import { Activity, Clock, GitBranch } from "lucide-react";

interface AnalyticsChartsProps {
  refreshKey?: number;
}

/**
 * Analytics & distribution panel (Masterplan §5.5).
 *
 * Derives three views from the audited history (GET /api/v1/history):
 *  1. Verdict mix — donut of AI vs. authentic.
 *  2. Confidence distribution — histogram of fake-probability bins.
 *  3. Latency trend — per-analysis processing latency over time.
 *
 * All values are computed from real backend history rows, not synthetic data.
 */
const AnalyticsCharts = ({ refreshKey = 0 }: AnalyticsChartsProps) => {
  const [history, setHistory] = useState<HistoryEntry[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        const rows = await getHistory(200);
        if (!cancelled) {
          setHistory(rows);
          setError(null);
        }
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : "History unavailable");
      }
    };
    load();
    return () => {
      cancelled = true;
    };
  }, [refreshKey]);

  const { verdictData, confidenceData, latencyData } = useMemo(() => {
    if (!history || history.length === 0) {
      return { verdictData: [], confidenceData: [], latencyData: [] };
    }

    const ai = history.filter((h) => h.verdict === "ai").length;
    const authentic = history.filter((h) => h.verdict === "authentic").length;

    const verdictData = [
      { name: "AI Generated", value: ai, color: "hsl(0 72% 51%)" },
      { name: "Authentic", value: authentic, color: "hsl(142 76% 45%)" },
      {
        name: "Uncertain",
        value: history.filter((h) => h.verdict === "uncertain").length,
        color: "hsl(38 92% 50%)",
      },
    ].filter((d) => d.value > 0);

    // Confidence histogram — 10 bins across [0, 100]%.
    const bins = Array.from({ length: 10 }, (_, i) => ({ bin: i, label: `${i * 10}-${i * 10 + 10}%`, count: 0 }));
    history.forEach((h) => {
      const pct = Math.min(9, Math.floor((h.confidence / 100) * 10));
      bins[pct].count += 1;
    });
    const confidenceData = bins;

    // Latency trend — chronological, capped at 60 points.
    const byTime = [...history]
      .sort((a, b) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime())
      .slice(-60)
      .map((h, i) => ({
        index: i,
        time: new Date(h.created_at).toLocaleTimeString(),
        latency: h.latency_ms,
      }));
    const latencyData = byTime;

    return { verdictData, confidenceData, latencyData };
  }, [history]);

  if (error) {
    return (
      <div className="rounded-xl border border-destructive/40 bg-destructive/5 p-4 text-sm text-destructive">
        {error} — is the FastAPI backend running on port 8000?
      </div>
    );
  }

  if (!history || history.length === 0) {
    return (
      <div className="glass rounded-xl p-6 text-center">
        <Activity className="h-8 w-8 mx-auto text-muted-foreground mb-2" />
        <p className="text-sm text-muted-foreground">
          No analysis history yet. Run a single analysis or a batch to populate the analytics.
        </p>
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
      {/* Verdict mix donut */}
      <div className="glass rounded-xl p-5">
        <div className="flex items-center gap-2 mb-3">
          <Activity className="h-4 w-4 text-primary" />
          <h3 className="text-sm font-semibold">Verdict Distribution</h3>
        </div>
        <ResponsiveContainer width="100%" height={180}>
          <PieChart>
            <Pie
              data={verdictData}
              dataKey="value"
              nameKey="name"
              cx="50%"
              cy="50%"
              innerRadius={45}
              outerRadius={70}
              paddingAngle={3}
              stroke="hsl(var(--card))"
            >
              {verdictData.map((entry) => (
                <Cell key={entry.name} fill={entry.color} />
              ))}
            </Pie>
            <Tooltip />
            <Legend
              verticalAlign="bottom"
              height={32}
              formatter={(value) => <span className="text-xs text-muted-foreground">{value}</span>}
            />
          </PieChart>
        </ResponsiveContainer>
      </div>

      {/* Confidence histogram */}
      <div className="glass rounded-xl p-5">
        <div className="flex items-center gap-2 mb-3">
          <GitBranch className="h-4 w-4 text-primary" />
          <h3 className="text-sm font-semibold">Confidence Distribution</h3>
        </div>
        <ResponsiveContainer width="100%" height={180}>
          <BarChart data={confidenceData}>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border) / 0.5)" />
            <XAxis dataKey="label" tick={{ fontSize: 9 }} interval={1} />
            <YAxis allowDecimals={false} tick={{ fontSize: 10 }} width={28} />
            <Tooltip />
            <Bar dataKey="count" name="Samples" fill="hsl(217 91% 60%)" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Latency trend */}
      <div className="glass rounded-xl p-5 md:col-span-2 xl:col-span-1">
        <div className="flex items-center gap-2 mb-3">
          <Clock className="h-4 w-4 text-primary" />
          <h3 className="text-sm font-semibold">Latency Trend (ms)</h3>
        </div>
        <ResponsiveContainer width="100%" height={180}>
          <LineChart data={latencyData}>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border) / 0.5)" />
            <XAxis dataKey="index" tick={{ fontSize: 10 }} interval="preserveStartEnd" />
            <YAxis tick={{ fontSize: 10 }} width={36} />
            <Tooltip
              labelFormatter={(label) => `Analysis #${Number(label) + 1}`}
              contentStyle={{ background: "hsl(var(--card))", border: "1px solid hsl(var(--border))" }}
            />
            <Line
              dataKey="latency"
              name="Latency"
              type="monotone"
              stroke="hsl(190 95% 50%)"
              dot={false}
              strokeWidth={2}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default AnalyticsCharts;
