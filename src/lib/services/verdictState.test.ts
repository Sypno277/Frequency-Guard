import { describe, it, expect } from "vitest";
import { resolveVerdictPresentation } from "./verdictState";

describe("resolveVerdictPresentation", () => {
  it("shows a confident AI label at high confidence", () => {
    const p = resolveVerdictPresentation("ai", 95, 0.9);
    expect(p.state).toBe("ai");
    expect(p.label).toBe("AI Generated");
    expect(p.tone).toBe("ai");
    expect(p.detail).toContain("AI-generation");
  });

  it("shows a confident Authentic label at low fake-probability", () => {
    // Low FAKE-confidence (8%) is strong REAL evidence, not uncertainty.
    const p = resolveVerdictPresentation("authentic", 8, 0.9);
    expect(p.state).toBe("authentic");
    expect(p.label).toBe("Authentic Image");
    expect(p.tone).toBe("real");
  });

  it("keeps an Authentic label even when fake-confidence is below 50", () => {
    // The backend verdict_state is authoritative; a low fake-probability
    // correctly means "real", which is NOT the same as "uncertain".
    const p = resolveVerdictPresentation("authentic", 45, 0.9);
    expect(p.state).toBe("authentic");
    expect(p.label).toBe("Authentic Image");
    expect(p.tone).toBe("real");
  });

  it("forces Uncertain when ensemble agreement is low", () => {
    const p = resolveVerdictPresentation("ai", 95, 0.4);
    expect(p.state).toBe("uncertain");
    expect(p.label).toBe("Uncertain");
    expect(p.tone).toBe("warning");
  });

  it("defaults a missing state to Uncertain", () => {
    const p = resolveVerdictPresentation("uncertain", 30, undefined);
    expect(p.state).toBe("uncertain");
    expect(p.label).toBe("Uncertain");
  });

  it("treats null agreement like absent agreement", () => {
    const p = resolveVerdictPresentation("ai", 87, null);
    expect(p.state).toBe("ai");
  });
});
