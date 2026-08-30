"""Verdict resolution: maps calibrated fake-probability to a truthful state.

Fixes the bug where an AI image was reported as "Uncertain" and then
"Authentic Image" at high confidence. That happened because the UI's
uncertainty gate (confidence < 50) and the model decision
(calibrated_fake >= threshold) were computed on different axes, and because
a stale threshold >= 0.5 could make "is_ai" essentially never true
(see reports/metrics_history.json: persisted thresholds of 1.0 and 0.001).

This module is the single source of truth for verdict semantics. A
"confident" label ("ai" / "authentic") is only emitted when the calibrated
confidence is >= 0.5 on the side it claims; otherwise the verdict is
"uncertain". This is E-Masterplan D0.1 / D2.2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

VERDICT_STATES = ("ai", "authentic", "uncertain")

#: How close (in calibrated fake-probability) a below-threshold prediction
#: can sit to the gate before we refuse to call it a confident "authentic".
_AMBIGUITY_MARGIN = 0.05
#: Absorbs float error in the ambiguity comparison (e.g. 0.4 - 0.05 = 0.3500...3).
_EPS = 1e-9

#: Hard lower bound for any usable AI-detection threshold.
_MIN_THRESHOLD = 0.05

#: Any threshold >= this value makes the AI gate effectively unreachable.
_MAX_USABLE_THRESHOLD = 0.5


@dataclass(frozen=True)
class VerdictResolution:
    """The resolved verdict for one prediction."""

    state: str  # "ai" | "authentic" | "uncertain"
    is_ai: bool  # legacy decision: calibrated_fake >= threshold
    confidence: float  # 0..100, calibrated

    @property
    def state_is_confident(self) -> bool:
        """True when the state is a confident label (not "uncertain")."""
        return self.state in ("ai", "authentic")


def clamp_threshold(raw: float | None) -> float:
    """Clamp an effective AI-detection threshold into [0.05, 0.5].

    A detector whose "fake" gate sits at >= 0.5 (or 1.0) is not a detector:
    it can never produce ``is_ai=True`` for any realistic input. We always
    fall back to a usable boundary rather than trust a corrupted persisted
    value (E-Masterplan D0.1).
    """
    if raw is None:
        return _MAX_USABLE_THRESHOLD
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return _MAX_USABLE_THRESHOLD
    # NaN/inf are not usable decision thresholds.
    if not math.isfinite(value):
        return _MAX_USABLE_THRESHOLD
    # Reject >= 0.5 entirely (would make the "AI" gate dead).
    if value >= _MAX_USABLE_THRESHOLD:
        return _MAX_USABLE_THRESHOLD
    return float(min(max(value, _MIN_THRESHOLD), _MAX_USABLE_THRESHOLD))


def resolve_verdict(calibrated_fake: float, threshold: float) -> VerdictResolution:
    """Resolve a verdict from a calibrated fake probability and decision threshold.

    Guarantees: a confident label is never emitted below the 0.5 confidence
    axis on the matching side. "uncertain" covers the ambiguous band around
    both the threshold and the 0.5 midpoint.
    """
    calibrated_fake = float(calibrated_fake)
    effective_threshold = clamp_threshold(threshold)
    is_ai = calibrated_fake >= effective_threshold
    confidence = calibrated_fake * 100.0

    if is_ai:
        # Above the gate but under the confident-claim axis -> ambiguous AI.
        state = "uncertain" if calibrated_fake < 0.5 else "ai"
    else:
        # Below the decision gate, but check proximity for ambiguity.
        if calibrated_fake >= effective_threshold - _AMBIGUITY_MARGIN - _EPS:
            # Too close to the gate to claim "authentic".
            state = "uncertain"
        else:
            state = "authentic"

    return VerdictResolution(
        state=state,
        is_ai=is_ai,
        confidence=confidence,
    )
