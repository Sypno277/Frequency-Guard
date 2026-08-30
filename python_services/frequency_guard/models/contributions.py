"""Per-feature contribution attribution (E-Masterplan E4.1).

Full SHAP would add a heavy dependency; for a 4-member sklearn ensemble we
get faithful per-feature contributions with a *grouped permutation*
strategy: for each candidate feature, measure how much the ensemble's fake
probability moves when that feature is replaced by its training median,
averaged over the members. This is exactly the "leave-one-feature-out"
attribution SHAP's independent variant approximates, at a fraction of the
cost (only the top-K display features are probed).

The output feeds the ``contributions`` field of the analyze response so the
dashboard can render horizontal contribution bars with tooltips.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .classifier import EnsembleClassifier

_EPS = 1e-12


@dataclass(frozen=True)
class FeatureContribution:
    """One feature's contribution to the current verdict."""

    feature: str
    value: float
    contribution: float  # signed push toward "fake" (positive) / "real" (negative)


@dataclass
class ContributionResult:
    """Top-K feature contributions for one sample."""

    items: list[FeatureContribution]
    member_probabilities: list[float]  # per-member fake probs (agreement meter)
    member_names: tuple[str, ...] = ("rf", "svm", "lr", "gb")

    @property
    def agreement(self) -> float:
        """Ensemble agreement: 1 = all members identical, 0 = maximally split.

        Computed as 1 - normalized std of member fake probabilities.
        """
        arr = np.asarray(self.member_probabilities, dtype=np.float64)
        if arr.size < 2:
            return 1.0
        return float(np.clip(1.0 - arr.std() / 0.5, 0.0, 1.0))


def compute_contributions(
    ensemble: EnsembleClassifier,
    feature_names: tuple[str, ...],
    vector: np.ndarray,
    top_k: int = 10,
    probe_features: int = 24,
) -> ContributionResult:
    """Attribute the current verdict to individual features.

    Args:
        ensemble: fitted ensemble classifier.
        feature_names: stable names aligned with the vector.
        vector: (1, n_features) feature matrix for one image.
        top_k: how many contributions to return.
        probe_features: how many top-variance features to probe (cost control).

    Returns:
        ContributionResult with signed contributions + member probabilities.
    """
    X = np.asarray(vector, dtype=np.float64).reshape(1, -1)
    n_features = X.shape[1]

    member_probs = ensemble.member_probabilities(X)[0, :, 1]
    baseline_fake = float(ensemble.predict_proba(X)[0, 1])

    # Probe the features with the largest |value| (they dominate the verdict
    # for tree/SVM members); cap the count to bound latency.
    probe_idx = np.argsort(-np.abs(X[0]))[: min(probe_features, n_features)]

    contributions: list[FeatureContribution] = []
    for idx in probe_idx:
        X_pert = X.copy()
        X_pert[0, idx] = 0.0  # neutralize (features are ~zero-mean post-norm)
        perturbed_fake = float(ensemble.predict_proba(X_pert)[0, 1])
        delta = baseline_fake - perturbed_fake  # positive = pushed toward fake
        contributions.append(
            FeatureContribution(
                feature=str(feature_names[idx]) if idx < len(feature_names) else f"f{idx}",
                value=round(float(X[0, idx]), 5),
                contribution=round(delta, 5),
            )
        )

    # Keep the largest-magnitude contributions, then sort by signed value.
    contributions.sort(key=lambda c: -abs(c.contribution))
    top = contributions[:top_k]
    top.sort(key=lambda c: -c.contribution)

    return ContributionResult(
        items=top,
        member_probabilities=[round(float(p), 4) for p in member_probs],
    )
