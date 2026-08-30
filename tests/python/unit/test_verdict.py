"""Unit tests for the verdict resolver (E-Masterplan D2.2)."""

from __future__ import annotations

import numpy as np
import pytest

from python_services.frequency_guard.models.verdict import (
    VERDICT_STATES,
    clamp_threshold,
    resolve_verdict,
)


class TestClampThreshold:
    def test_usable_threshold_passthrough(self) -> None:
        assert clamp_threshold(0.37) == pytest.approx(0.37)
        assert clamp_threshold(0.08) == pytest.approx(0.08)

    def test_rejects_near_unity(self) -> None:
        assert clamp_threshold(0.5) == 0.5
        assert clamp_threshold(0.6) == 0.5
        assert clamp_threshold(0.9) == 0.5
        assert clamp_threshold(1.0) == 0.5

    def test_clamps_low_to_min(self) -> None:
        assert clamp_threshold(0.0) == pytest.approx(0.05)
        assert clamp_threshold(0.001) == pytest.approx(0.05)

    def test_none_defaults_half(self) -> None:
        assert clamp_threshold(None) == 0.5

    def test_non_numeric_defaults_half(self) -> None:
        assert clamp_threshold("bad") == 0.5


class TestResolveVerdict:
    def test_states_exhaustive(self) -> None:
        assert set(VERDICT_STATES) == {"ai", "authentic", "uncertain"}

    def test_ai_when_high_fake_and_above_threshold(self) -> None:
        v = resolve_verdict(0.95, 0.5)
        assert v.state == "ai"
        assert v.is_ai is True
        assert v.confidence == pytest.approx(95.0)

    def test_authentic_when_low_fake_below_threshold(self) -> None:
        v = resolve_verdict(0.05, 0.5)
        assert v.state == "authentic"
        assert v.is_ai is False
        assert v.confidence == pytest.approx(5.0)

    def test_uncertain_when_above_gate_but_below_half(self) -> None:
        v = resolve_verdict(0.3, 0.08)
        assert v.state == "uncertain"
        assert v.is_ai is True  # model says AI, but confidence is < 50
        assert v.confidence == pytest.approx(30.0)

    def test_uncertain_when_close_to_gate_below(self) -> None:
        # Below the gate but within the ambiguity margin -> uncertain.
        v = resolve_verdict(0.35, 0.4)
        assert v.state == "uncertain"
        assert v.is_ai is False

    def test_authentic_when_confidently_below_gate(self) -> None:
        v = resolve_verdict(0.2, 0.4)
        assert v.state == "authentic"
        assert v.is_ai is False

    def test_confident_label_never_below_half(self) -> None:
        for fake in np.linspace(0.0, 0.99, 200):
            v = resolve_verdict(float(fake), 0.5)
            if v.state == "ai":
                assert v.confidence >= 50.0
            if v.state == "authentic":
                assert v.confidence < 50.0

    def test_stale_threshold_does_not_disable_ai(self) -> None:
        # Even a stale 1.0 threshold must not turn a strong AI signal into
        # "authentic" — clamp_threshold prevents the gate from being dead.
        v = resolve_verdict(0.95, 1.0)
        assert v.state == "ai"
        assert v.is_ai is True
