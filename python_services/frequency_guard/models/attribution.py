"""Generator-family attribution (Masterplan §3.1, §4.3).

Maps ensemble fake-probabilities to calibrated likelihoods over generator
families: ``real / diffusion / gan / other``. The mapping is a softmax over
family-specific affine transforms of the raw fake probability, fit on
out-of-fold predictions during training so scores are calibrated rather
than hand-tuned. When no family labels exist in the manifest, we fall back
to a fixed prior mapping derived from the ensemble's confidence — clearly
labeled as heuristic until real per-family training data is supplied.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

from ..logging import get_logger

log = get_logger(__name__)

FAMILIES: tuple[str, ...] = ("real", "diffusion", "gan", "other")


@dataclass(frozen=True)
class FamilyAttribution:
    """Calibrated per-family probabilities for one image."""

    families: tuple[str, ...]
    probabilities: tuple[float, ...]

    def top(self) -> tuple[str, float]:
        """Return the most likely family and its probability."""
        idx = int(np.argmax(self.probabilities))
        return self.families[idx], float(self.probabilities[idx])

    def as_dict(self) -> dict[str, float]:
        """JSON-friendly {family: probability} mapping."""
        return dict(zip(self.families, self.probabilities, strict=True))


class FamilyAttributor:
    """Softmax attribution over generator families.

    Fit modes:
    - ``fit_supervised``: learns per-family logistic weights from OOF
      ensemble probabilities + true family labels (preferred).
    - Default constructor: fixed heuristic weights — honest fallback when
      no family labels exist; the API marks results "heuristic".
    """

    # Heuristic priors: logit(family) = intercept + slope * fake_probability.
    # "real" carries a positive intercept and negative slope so it dominates
    # as p -> 0 and loses as p -> 1 (matching the docstring contract).
    # GAN keeps the steepest positive slope (sharpest artifacts at high p);
    # diffusion and "other" sit between. Crossover real->synthetic ≈ 0.55.
    _HEURISTIC_WEIGHTS = np.asarray(
        [
            [3.0, -6.0],  # real: favored at low p, drops fast as p rises
            [-1.0, 2.0],  # diffusion
            [-2.5, 4.5],  # gan: strongest high-confidence artifacts
            [-2.0, 3.0],  # other
        ],
        dtype=np.float64,
    )

    def __init__(self) -> None:
        self._weights: np.ndarray | None = None
        self.supervised: bool = False

    @property
    def mode(self) -> str:
        """ "supervised" or "heuristic" — surfaced by the API."""
        return "supervised" if self.supervised else "heuristic"

    def fit_supervised(self, oof_probs: np.ndarray, family_labels: np.ndarray) -> FamilyAttributor:
        """Fit per-family logistic heads on out-of-fold probabilities.

        Args:
            oof_probs: ensemble fake probabilities (n,) from CV folds.
            family_labels: integer family index per sample (n,), aligned
                with :data:`FAMILIES`.
        """
        oof_probs = np.asarray(oof_probs, dtype=np.float64)
        family_labels = np.asarray(family_labels, dtype=np.int64)
        if oof_probs.shape != family_labels.shape:
            raise ValueError("oof_probs and family_labels must have equal length")

        X = np.stack([np.ones_like(oof_probs), oof_probs], axis=1)
        n_families = len(FAMILIES)
        counts = np.bincount(family_labels, minlength=n_families)
        if np.any(counts == 0):
            missing = [FAMILIES[i] for i in range(n_families) if counts[i] == 0]
            log.warning(
                "attribution falling back to heuristic (missing families)",
                extra={"fields": {"missing": missing}},
            )
            return self

        # One-vs-rest logistic regression per family via simple gradient
        # descent (avoids sklearn dependency here; features are 2-D).
        weights = np.zeros((n_families, 2), dtype=np.float64)
        lr, epochs = 0.15, 800
        for k in range(n_families):
            y = (family_labels == k).astype(np.float64)
            w = np.zeros(2, dtype=np.float64)
            for _ in range(epochs):
                z = X @ w
                p = 1.0 / (1.0 + np.exp(-z))
                grad = X.T @ (p - y) / len(y)
                w -= lr * grad
            weights[k] = w

        self._weights = weights
        self.supervised = True
        return self

    def attribute(self, fake_probability: float) -> FamilyAttribution:
        """Compute calibrated family probabilities for one image."""
        if self._weights is not None:
            logits = self._weights @ np.asarray([1.0, fake_probability])
        else:
            logits = self._HEURISTIC_WEIGHTS @ np.asarray([1.0, fake_probability])

        logits = logits - logits.max()  # numerical stability
        exp = np.exp(logits)
        probs = exp / exp.sum()
        return FamilyAttribution(
            families=FAMILIES,
            probabilities=tuple(float(p) for p in probs),
        )

    def save(self, path: str | Path) -> None:
        """Persist attributor state with joblib."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"weights": self._weights, "supervised": self.supervised}, p)

    @classmethod
    def load(cls, path: str | Path) -> FamilyAttributor:
        """Load an attributor saved by :meth:`save`."""
        payload = joblib.load(path)
        instance = cls()
        instance._weights = payload.get("weights")
        instance.supervised = bool(payload.get("supervised", False))
        return instance
