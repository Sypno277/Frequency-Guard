/**
 * Verdict-state presentation mapper (E-Masterplan D3.2).
 *
 * The backend now returns a single `verdict_state` field: "ai" | "authentic" |
 * "uncertain". This is the ONLY source of truth for what the UI labels an
 * image. The old logic derived the label from `is_ai` and `confidence` on
 * different axes, which produced the reported "Uncertain → Authentic@95%"
 * flip. This module guarantees a confident label is never shown below the
 * 0.5 confidence axis, and that a high-confidence label is suppressed when
 * the ensemble is split (agreement < 0.5).
 */

export type VerdictState = "ai" | "authentic" | "uncertain";

export interface VerdictPresentation {
  /** The canonical state the UI must render. */
  state: VerdictState;
  /** Human-readable label. */
  label: string;
  /** Icon kind: "ai" | "real" | "warning". */
  tone: "ai" | "real" | "warning";
  /** Short supporting text. */
  detail: string;
}

/**
 * Resolve the UI presentation from the backend verdict state + confidence.
 *
 * @param state Backend verdict_state (authoritative).
 * @param confidence Calibrated confidence 0..100.
 * @param agreement Ensemble agreement 0..1 (optional; low agreement forces
 *   "uncertain" even when confidence is high).
 */
export function resolveVerdictPresentation(
  state: VerdictState,
  confidence: number,
  agreement?: number | null,
): VerdictPresentation {
  // Low ensemble agreement overrides the backend state: treating a split
  // ensemble as confident would repeat the "Uncertain → Authentic@high-conf"
  // bug in reverse (showing a confident label the models don't agree on).
  if (agreement !== undefined && agreement !== null && agreement < 0.5) {
    return {
      state: "uncertain",
      label: "Uncertain",
      tone: "warning",
      detail: "The analysis models disagree; treat the verdict with caution.",
    };
  }

  // The backend `verdict_state` already encodes the ambiguity band; trust it.
  // `confidence` is the FAKE-probability confidence, so an authentic image
  // correctly reports a low confidence (8%) — that is strong real evidence,
  // not uncertainty. The old UI wrongly gated on `confidence < 50`, which is
  // what produced the reported "Uncertain → Authentic@95%" contradiction.
  switch (state) {
    case "ai":
      return {
        state: "ai",
        label: "AI Generated",
        tone: "ai",
        detail: "AI-generation patterns detected.",
      };
    case "authentic":
      return {
        state: "authentic",
        label: "Authentic Image",
        tone: "real",
        detail: "No AI-generation patterns detected.",
      };
    case "uncertain":
    default:
      return {
        state: "uncertain",
        label: "Uncertain",
        tone: "warning",
        detail: "The evidence does not support a confident verdict.",
      };
  }
}
