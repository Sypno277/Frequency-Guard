import { useCallback, useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { getHistory, type HistoryEntry } from "@/lib/api/client";
import { RefreshCw, History as HistoryIcon } from "lucide-react";

interface HistoryPanelProps {
  /** Increment this to force a refresh (e.g., after a new analysis). */
  refreshKey?: number;
  limit?: number;
}

/**
 * History / audit log (Masterplan §5.5) — past analyses with timestamp,
 * verdict, confidence, latency, model version and request id. Data comes
 * from the SQLite-backed GET /api/v1/history endpoint.
 */
const HistoryPanel = ({ refreshKey = 0, limit = 100 }: HistoryPanelProps) => {
  const [rows, setRows] = useState<HistoryEntry[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isRefreshing, setIsRefreshing] = useState(false);

  const load = useCallback(async () => {
    setIsRefreshing(true);
    try {
      const data = await getHistory(limit);
      setRows(data);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "History unavailable");
    } finally {
      setIsRefreshing(false);
    }
  }, [limit]);

  useEffect(() => {
    load();
  }, [load, refreshKey]);

  return (
    <div className="glass rounded-xl p-6">
      <div className="mb-4 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <HistoryIcon className="h-5 w-5 text-primary" />
          <h3 className="text-lg font-semibold">Analysis History</h3>
        </div>
        <Button variant="outline" size="sm" onClick={load} disabled={isRefreshing}>
          <RefreshCw className={`h-4 w-4 ${isRefreshing ? "animate-spin" : ""}`} />
          Refresh
        </Button>
      </div>

      {error && <p className="text-sm text-destructive">{error} — is the API running?</p>}

      {rows && rows.length === 0 && (
        <p className="text-sm text-muted-foreground">No analyses recorded yet.</p>
      )}

      {rows && rows.length > 0 && (
        <div className="rounded-lg border border-border overflow-x-auto max-h-[420px] overflow-y-auto">
          <table className="w-full text-sm">
            <thead className="bg-muted/30 text-left sticky top-0">
              <tr>
                <th className="px-3 py-2">Time</th>
                <th className="px-3 py-2">Source</th>
                <th className="px-3 py-2">Verdict</th>
                <th className="px-3 py-2">Confidence</th>
                <th className="px-3 py-2">Latency</th>
                <th className="px-3 py-2">Model</th>
                <th className="px-3 py-2">Request</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.id} className="border-t border-border hover:bg-muted/20">
                  <td className="px-3 py-2 whitespace-nowrap">{new Date(row.created_at).toLocaleString()}</td>
                  <td className="px-3 py-2 max-w-[180px] truncate" title={row.source}>
                    {row.source}
                  </td>
                  <td className="px-3 py-2">
                    {row.verdict === "ai" ? (
                      <Badge variant="destructive">AI</Badge>
                    ) : (
                      <Badge variant="secondary">Authentic</Badge>
                    )}
                  </td>
                  <td className="px-3 py-2 font-mono">{row.confidence.toFixed(1)}%</td>
                  <td className="px-3 py-2 font-mono">{row.latency_ms.toFixed(0)} ms</td>
                  <td className="px-3 py-2 font-mono text-xs">v{row.model_version}</td>
                  <td className="px-3 py-2 font-mono text-xs text-muted-foreground">{row.request_id}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};

export default HistoryPanel;
