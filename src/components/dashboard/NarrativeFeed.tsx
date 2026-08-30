import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { Newspaper } from "lucide-react";
import { getHistory } from "@/lib/api/client";
import type { HistoryEntry } from "@/lib/api/client";

interface NarrativeFeedProps {
  refreshKey?: number;
  limit?: number;
}

const SEVERITY_STYLES: Record<string, { text: string; bg: string; label: string }> = {
  "[DETERMINISTIC AI]": { text: "text-destructive", bg: "bg-destructive/20", label: "Deterministic AI" },
  "[PROBABLE AI]": { text: "text-warning", bg: "bg-warning/20", label: "Probable AI" },
  "[INCONCLUSIVE]": { text: "text-muted-foreground", bg: "bg-muted/30", label: "Inconclusive" },
  "[DETERMINISTIC REAL]": { text: "text-success", bg: "bg-success/20", label: "Deterministic Real" },
  "[PROVENANCE VERIFIED]": { text: "text-primary", bg: "bg-primary/20", label: "Provenance Verified" },
};

const fallbackStyle = { text: "text-muted-foreground", bg: "bg-muted/30", label: "Analysis" };

/**
 * FrequencyGuard Phase 5 — Linguistic Feed.
 *
 * Real-time narrative summaries of recent analyses, rendered in plain
 * English with a severity prefix. Every sentence is grounded in the
 * actual measured signals from the backend; operators read *why* an
 * image was flagged without parsing heatmaps or confidence numbers.
 */
const NarrativeFeed = ({ refreshKey = 0, limit = 12 }: NarrativeFeedProps) => {
  const [rows, setRows] = useState<HistoryEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      setLoading(true);
      setError(null);
      try {
        const data = await getHistory(limit);
        if (!cancelled) setRows(data);
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : "Failed to load narrative feed");
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    load();
    return () => {
      cancelled = true;
    };
  }, [refreshKey, limit]);

  return (
    <div className="glass rounded-xl p-6">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-lg font-semibold flex items-center gap-2">
            <Newspaper className="h-5 w-5 text-primary" />
            Linguistic Narrative Feed
          </h3>
          <p className="text-sm text-muted-foreground mt-1">
            Auto-synthesized summaries from the NLP engine — each sentence is grounded in measured
            signals.
          </p>
        </div>
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <span className="h-2 w-2 rounded-full bg-success animate-pulse" />
          Live
        </div>
      </div>

      {loading && <div className="text-sm text-muted-foreground py-8 text-center">Loading narrative feed…</div>}
      {error && <div className="text-sm text-destructive py-8 text-center">{error}</div>}

      {!loading && !error && (
        <div className="space-y-3">
          {rows.length === 0 && (
            <div className="text-sm text-muted-foreground py-6 text-center">
              No analyses yet — upload an image to populate the feed.
            </div>
          )}
          {rows.map((row, i) => {
            const severity = row.severity ?? "";
            const style = SEVERITY_STYLES[severity] ?? fallbackStyle;
            const hasNarrative = row.narrative && row.narrative.length > 0;
            return (
              <motion.div
                key={row.id}
                className="rounded-lg border border-border bg-muted/20 p-4"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.3, delay: i * 0.04 }}
              >
                <div className="flex flex-wrap items-center gap-2 mb-2">
                  <span
                    className={`inline-flex px-2 py-0.5 rounded-full text-xs font-medium ${style.text} ${style.bg}`}
                  >
                    {style.label}
                  </span>
                  <span className="text-xs text-muted-foreground font-mono">{row.source}</span>
                  <span className="text-xs text-muted-foreground">
                    {new Date(row.created_at).toLocaleString()}
                  </span>
                  <span className="ml-auto text-xs text-muted-foreground">
                    conf {row.confidence.toFixed(0)}%
                  </span>
                </div>
                <p className="text-sm text-foreground leading-relaxed">
                  {hasNarrative ? row.narrative : `Verdict: ${row.verdict_state ?? row.verdict}.`}
                </p>
              </motion.div>
            );
          })}
        </div>
      )}
    </div>
  );
};

export default NarrativeFeed;
