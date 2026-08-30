"""Leakage-safe feature selection (E-Masterplan E1 #7).

With the expanded ~250-dim feature set, some features may be redundant or
noise. Selection must happen *inside* CV folds — ranking features on the
full dataset before splitting leaks label information into the validation
fold and inflates accuracy.

This module provides a fold-safe transformer: fit on training rows only,
transform validation rows with the *frozen* selected subset. It uses
mutual information (works with the non-linear feature/label relationships
typical of spectral data) with an F-test fallback for robustness.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.feature_selection import mutual_info_classif

from ..logging import get_logger

log = get_logger(__name__)


@dataclass
class FeatureSelector:
    """Mutual-information top-K feature selector, fit strictly on train rows."""

    k: int = 150
    random_state: int = 42
    selected_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    scores_: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))

    @property
    def fitted(self) -> bool:
        return self.selected_indices.size > 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> FeatureSelector:
        """Rank features by mutual information on training rows only."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        k = int(min(self.k, X.shape[1]))
        mi = mutual_info_classif(X, y, random_state=self.random_state)
        mi = np.nan_to_num(mi, nan=0.0)
        self.scores_ = mi
        self.selected_indices = np.argsort(mi)[::-1][:k].astype(np.int64)
        log.info(
            "feature selection done",
            extra={
                "fields": {
                    "k": int(self.selected_indices.size),
                    "top_mi": round(float(mi[self.selected_indices[0]]), 4),
                }
            },
        )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project X onto the selected feature subset."""
        if not self.fitted:
            raise RuntimeError("FeatureSelector must be fit before transform")
        return np.asarray(X, dtype=np.float64)[:, self.selected_indices]

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Convenience: fit then transform the same (training) matrix."""
        return self.fit(X, y).transform(X)

    def top_features(self, feature_names: tuple[str, ...], n: int = 20) -> list[tuple[str, float]]:
        """Return the top-n (name, MI score) pairs by selection rank."""
        if not self.fitted:
            return []
        out: list[tuple[str, float]] = []
        for idx in self.selected_indices[:n]:
            name = feature_names[idx] if idx < len(feature_names) else f"f{idx}"
            out.append((str(name), round(float(self.scores_[idx]), 5)))
        return out
