"""Calibrated classical-ML ensemble for real-vs-AI classification.

Four diverse scikit-learn learners are blended by probability averaging:
- RandomForest (non-linear, robust to noise)
- SVM with RBF kernel (margin-based, good on mid-dimensional features)
- LogisticRegression (linear baseline, well-scaled probabilities)
- GradientBoosting (sequential additive model)

The probability-average ensemble is then wrapped by a calibration layer
(see ``models/calibration.py``) and, optionally, a stacked logistic
meta-learner trained on out-of-fold predictions (see
``training/train_pipeline.py``). This class is CPU-only and fast: single
images classify in well under 5ms.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from ..logging import get_logger

log = get_logger(__name__)

FAKE_INDEX = 1  # class column order: 0 = real, 1 = fake


@dataclass
class EnsembleConfig:
    """Hyperparameters for the four ensemble members."""

    rf_trees: int = 300
    rf_max_depth: int | None = 24
    gb_estimators: int = 200
    gb_max_depth: int = 4
    gb_learning_rate: float = 0.05
    svm_c: float = 4.0
    svm_gamma: str = "scale"
    lr_max_iter: int = 2000
    random_state: int = 42
    n_jobs: int = -1


class EnsembleClassifier:
    """Blend of RandomForest, SVM-RBF, LogisticRegression, GradientBoost."""

    def __init__(self, config: EnsembleConfig | None = None) -> None:
        self.config = config or EnsembleConfig()
        self.feature_names: tuple[str, ...] | None = None

        self.rf = RandomForestClassifier(
            n_estimators=self.config.rf_trees,
            max_depth=self.config.rf_max_depth,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
        )
        self.svm = SVC(
            C=self.config.svm_c,
            gamma=self.config.svm_gamma,
            probability=True,
            random_state=self.config.random_state,
        )
        self.lr = LogisticRegression(
            max_iter=self.config.lr_max_iter,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
        )
        self.gb = GradientBoostingClassifier(
            n_estimators=self.config.gb_estimators,
            max_depth=self.config.gb_max_depth,
            learning_rate=self.config.gb_learning_rate,
            random_state=self.config.random_state,
        )
        self._fitted = False

    @property
    def fitted(self) -> bool:
        return self._fitted

    def fit(self, X: np.ndarray, y: np.ndarray) -> EnsembleClassifier:
        """Train all four members on the feature matrix ``X`` and labels ``y``."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        if len(np.unique(y)) != 2:
            raise ValueError("Ensemble requires exactly two classes in y")

        self.rf.fit(X, y)
        self.svm.fit(X, y)
        self.lr.fit(X, y)
        self.gb.fit(X, y)
        self._fitted = True
        log.info(
            "ensemble trained",
            extra={
                "fields": {
                    "n_samples": int(X.shape[0]),
                    "n_features": int(X.shape[1]),
                }
            },
        )
        return self

    def member_probabilities(self, X: np.ndarray) -> np.ndarray:
        """Return (n_samples, 4, 2) per-member probabilities [real, fake]."""
        X = np.asarray(X, dtype=np.float64)
        return np.stack(
            [
                self.rf.predict_proba(X),
                self.svm.predict_proba(X),
                self.lr.predict_proba(X),
                self.gb.predict_proba(X),
            ],
            axis=1,
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Probability-average ensemble predictions (n,2): [real, fake]."""
        if not self._fitted:
            raise RuntimeError("EnsembleClassifier must be fit before predict_proba")
        member = self.member_probabilities(X)
        return member.mean(axis=1)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Hard labels (0=real, 1=fake) using ``threshold`` on fake probability."""
        proba = self.predict_proba(X)
        return (proba[:, FAKE_INDEX] >= threshold).astype(np.int64)

    def save(self, path: str | Path) -> None:
        """Persist the ensemble and its feature names with joblib."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": self.config,
            "feature_names": self.feature_names,
            "rf": self.rf,
            "svm": self.svm,
            "lr": self.lr,
            "gb": self.gb,
        }
        joblib.dump(payload, p)
        log.info("ensemble saved", extra={"fields": {"path": str(p)}})

    @classmethod
    def load(cls, path: str | Path) -> EnsembleClassifier:
        """Load an ensemble persisted by :meth:`save`."""
        payload: dict[str, Any] = joblib.load(path)
        instance = cls(config=payload["config"])
        instance.rf = payload["rf"]
        instance.svm = payload["svm"]
        instance.lr = payload["lr"]
        instance.gb = payload["gb"]
        instance.feature_names = payload.get("feature_names")
        instance._fitted = True
        return instance


class StackedMetaClassifier:
    """Logistic meta-learner over the four base learners' fake probabilities.

    Masterplan §4.3: instead of uniform probability averaging, a logistic
    regression learns optimal per-member weights from out-of-fold member
    predictions (fit in ``training/train_pipeline.py``), avoiding leakage.
    Until ``fit`` is called the learner is inert; callers fall back to
    plain averaging.
    """

    def __init__(self, c_reg: float = 1.0) -> None:
        self.c_reg = c_reg
        self._model: LogisticRegression | None = None

    @property
    def fitted(self) -> bool:
        return self._model is not None

    def _fake_features(self, member_probs: np.ndarray) -> np.ndarray:
        """Project (n, 4, 2) member probabilities to the (n, 4) fake column."""
        arr = np.asarray(member_probs, dtype=np.float64)
        if arr.ndim != 3 or arr.shape[1:] != (4, 2):
            raise ValueError(f"expected (n, 4, 2) member probabilities, got shape {arr.shape}")
        return arr[:, :, FAKE_INDEX]

    def fit(self, member_probs: np.ndarray, y: np.ndarray) -> StackedMetaClassifier:
        """Fit the logistic meta-head on out-of-fold member probabilities."""
        X = self._fake_features(member_probs)
        y_arr = np.asarray(y, dtype=np.int64)
        if len(np.unique(y_arr)) != 2:
            raise ValueError("StackedMetaClassifier requires exactly two classes")
        self._model = LogisticRegression(C=self.c_reg, max_iter=2000, random_state=0)
        self._model.fit(X, y_arr)
        coef = [round(float(v), 4) for v in self._model.coef_[0]]
        log.info("stacked meta-learner trained", extra={"fields": {"coef_rf_svm_lr_gb": coef}})
        return self

    def predict_proba(self, member_probs: np.ndarray) -> np.ndarray:
        """Return (n, 2) stacked probabilities."""
        if self._model is None:
            raise RuntimeError("StackedMetaClassifier must be fit before predict_proba")
        return self._model.predict_proba(self._fake_features(member_probs))

    def predict_fake(self, member_probs: np.ndarray) -> np.ndarray:
        """Return the (n,) stacked fake-class probability."""
        return self.predict_proba(member_probs)[:, FAKE_INDEX]

    def save(self, path: str | Path) -> None:
        """Persist the meta-learner with joblib."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"c_reg": self.c_reg, "model": self._model}, p)
        log.info("stacked meta-learner saved", extra={"fields": {"path": str(p)}})

    @classmethod
    def load(cls, path: str | Path) -> StackedMetaClassifier:
        """Load a meta-learner persisted by :meth:`save`."""
        payload: dict[str, Any] = joblib.load(path)
        instance = cls(c_reg=payload.get("c_reg", 1.0))
        instance._model = payload["model"]
        return instance
