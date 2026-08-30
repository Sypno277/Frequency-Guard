"""Calibration of ensemble probabilities.

Wraps the blended probability with a post-hoc calibrator (isotonic
regression by default, sigmoid/Platt as an alternative). Calibration maps
raw ensemble probability to a well-calibrated confidence, and we report the
Expected Calibration Error (ECE) so accuracy claims are grounded.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from ..logging import get_logger

log = get_logger(__name__)


@dataclass
class CalibrationResult:
    """Metrics from fitting a calibrator."""

    ece_before: float
    ece_after: float
    brier_before: float
    brier_after: float


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    """Compute the Expected Calibration Error over confidence bins.

    Bins are equal-width in predicted fake probability. ECE is the weighted
    mean of |accuracy - confidence| across bins.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    if y_true.shape != y_prob.shape:
        raise ValueError("y_true and y_prob must have the same shape")

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if not np.any(mask):
            continue
        conf = float(np.mean(y_prob[mask]))
        acc = float(np.mean(y_true[mask] == 1))
        ece += np.abs(conf - acc) * np.mean(mask)
    return float(ece)


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Brier score (mean squared error of predicted probability)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    return float(np.mean((y_true - y_prob) ** 2))


class ProbabilityCalibrator:
    """Map raw fake-probability to a calibrated confidence.

    Supports isotonic regression (monotonic, flexible) or sigmoid (Platt)
    calibration fit on held-out data via ``fit``.
    """

    def __init__(self, method: str = "isotonic") -> None:
        if method not in ("isotonic", "sigmoid"):
            raise ValueError(f"method must be 'isotonic' or 'sigmoid', got '{method}'")
        self.method = method
        self._model: Any = None
        self._fitted = False

    def fit(self, raw_prob: np.ndarray, y_true: np.ndarray) -> ProbabilityCalibrator:
        """Fit the calibrator on raw probabilities ``raw_prob`` and labels.

        Args:
            raw_prob: fake-class probabilities from the ensemble (n,).
            y_true: binary labels (0=real, 1=fake).

        Returns:
            self
        """
        raw_prob = np.asarray(raw_prob, dtype=np.float64)
        y_true = np.asarray(y_true, dtype=np.int64)
        if raw_prob.shape != y_true.shape:
            raise ValueError("raw_prob and y_true must be same length")

        if self.method == "isotonic":
            try:
                self._model = IsotonicRegression(out_of_bounds="clip")
                self._model.fit(raw_prob, y_true)
                # Overfit guard (E-Masterplan D2.1): if the isotonic fit is a
                # near-step (>=95% of mapped outputs at exactly {0,1}), it is
                # overfit to a bimodal synthetic distribution and will collapse
                # an unseen mid-range score (e.g. a real AI image's 0.3-0.4)
                # down to ~0.0-0.05. Fall back to Platt/sigmoid, which is
                # monotone and never saturates so aggressively.
                mapped = np.clip(self._model.predict(raw_prob), 0.0, 1.0)
                extreme = float(np.mean((mapped <= 1e-6) | (mapped >= 1.0 - 1e-6)))
                if extreme >= 0.95:
                    log.warning(
                        "calibrator_overfit: isotonic mapped %.0f%% of scores to {0,1}; "
                        "falling back to sigmoid",
                        extreme * 100,
                    )
                    self._fit_sigmoid(raw_prob, y_true)
            except ValueError:
                # Degenerate single-class input; fall back gracefully.
                self._fit_sigmoid(raw_prob, y_true)
        else:
            self._fit_sigmoid(raw_prob, y_true)

        self._fitted = True
        return self

    def _fit_sigmoid(self, raw_prob: np.ndarray, y_true: np.ndarray) -> None:
        """Platt scaling: logistic regression on logit(prob)."""
        # Record the actual method so predict_proba/save/load route correctly.
        self.method = "sigmoid"
        logit = np.log(np.clip(raw_prob, 1e-7, 1 - 1e-7) / (1 - np.clip(raw_prob, 1e-7, 1 - 1e-7)))
        X = logit.reshape(-1, 1)
        base = LogisticRegression()
        self._model = CalibratedClassifierCV(base, method="sigmoid", cv=3)
        self._model.fit(X, y_true)

    def predict_proba(self, raw_prob: np.ndarray) -> np.ndarray:
        """Return calibrated fake probabilities (n,)."""
        raw_prob = np.asarray(raw_prob, dtype=np.float64)
        if not self._fitted:
            raise RuntimeError("ProbabilityCalibrator must be fit before predict_proba")
        if self.method == "isotonic":
            return self._model.predict(raw_prob)
        logit = np.log(np.clip(raw_prob, 1e-7, 1 - 1e-7) / (1 - np.clip(raw_prob, 1e-7, 1 - 1e-7)))
        return self._model.predict_proba(logit.reshape(-1, 1))[:, 1]

    def evaluate(self, raw_prob: np.ndarray, y_true: np.ndarray) -> CalibrationResult:
        """Fit-independent evaluation: ECE and Brier before/after calibration."""
        calibrated = self.predict_proba(raw_prob)
        return CalibrationResult(
            ece_before=expected_calibration_error(y_true, raw_prob),
            ece_after=expected_calibration_error(y_true, calibrated),
            brier_before=brier_score(y_true, raw_prob),
            brier_after=brier_score(y_true, calibrated),
        )

    def save(self, path: str | Path) -> None:
        """Persist the calibrator with joblib."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"method": self.method, "model": self._model}, p)

    @classmethod
    def load(cls, path: str | Path) -> ProbabilityCalibrator:
        """Load a calibrator saved by :meth:`save`."""
        payload: dict[str, Any] = joblib.load(path)
        instance = cls(method=payload["method"])
        instance._model = payload["model"]
        instance._fitted = True
        return instance
