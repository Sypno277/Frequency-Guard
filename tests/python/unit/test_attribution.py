"""Unit tests for generator-family attribution."""

from __future__ import annotations

import numpy as np
import pytest

from python_services.frequency_guard.models.attribution import FAMILIES, FamilyAttribution, FamilyAttributor


class TestFamilyAttribution:
    def test_top_returns_max_family(self) -> None:
        attr = FamilyAttribution(families=FAMILIES, probabilities=(0.1, 0.7, 0.15, 0.05))
        family, prob = attr.top()
        assert family == "diffusion"
        assert prob == pytest.approx(0.7)

    def test_as_dict_maps_families(self) -> None:
        attr = FamilyAttribution(families=FAMILIES, probabilities=(0.6, 0.2, 0.1, 0.1))
        d = attr.as_dict()
        assert set(d.keys()) == set(FAMILIES)
        assert d["real"] == pytest.approx(0.6)


class TestFamilyAttributor:
    def test_default_is_heuristic_mode(self) -> None:
        attr = FamilyAttributor()
        assert attr.mode == "heuristic"
        assert attr._weights is None

    def test_heuristic_attribution_probabilities_sum_to_one(self) -> None:
        attr = FamilyAttributor()
        out = attr.attribute(0.8)
        assert len(out.probabilities) == 4
        assert sum(out.probabilities) == pytest.approx(1.0, abs=1e-6)
        # High fake probability should favor a synthetic family over real.
        top_family, _ = out.top()
        assert top_family in ("diffusion", "gan", "other")

    def test_heuristic_low_probability_favors_real(self) -> None:
        attr = FamilyAttributor()
        out = attr.attribute(0.05)
        top_family, _ = out.top()
        assert top_family == "real"

    def test_fit_supervised_requires_match_length(self) -> None:
        attr = FamilyAttributor()
        with pytest.raises(ValueError):
            attr.fit_supervised(np.asarray([0.1, 0.2]), np.asarray([0, 1, 2]))

    def test_fit_supervised_missing_family_falls_back(self) -> None:
        """If some families are absent from the labels, keep heuristic mode."""
        attr = FamilyAttributor()
        oof = np.asarray([0.1, 0.2, 0.8, 0.9])
        labels = np.asarray([0, 0, 1, 1])  # no gan/other
        attr.fit_supervised(oof, labels)
        assert attr.mode == "heuristic"
        assert attr._weights is None

    def test_fit_supervised_all_families_present(self) -> None:
        attr = FamilyAttributor()
        rng = np.random.default_rng(11)
        n = 80
        oof = rng.uniform(0.0, 1.0, n)
        labels = rng.integers(0, len(FAMILIES), n)
        # Force all families present:
        labels[: len(FAMILIES)] = np.arange(len(FAMILIES))
        attr.fit_supervised(oof, labels)
        assert attr.mode == "supervised"
        assert attr._weights is not None

    def test_attribution_shifts_with_high_fake_prob(self) -> None:
        """Supervised attribution should give a high fake prob a synthetic top family."""
        attr = FamilyAttributor()
        rng = np.random.default_rng(12)
        n = 120
        oof = np.concatenate(
            [
                rng.uniform(0.0, 0.2, n // 4),  # real
                rng.uniform(0.6, 0.9, n // 4),  # diffusion
                rng.uniform(0.7, 0.95, n // 4),  # gan
                rng.uniform(0.5, 0.8, n // 4),  # other
            ]
        )
        labels = np.asarray([0] * (n // 4) + [1] * (n // 4) + [2] * (n // 4) + [3] * (n // 4))
        attr.fit_supervised(oof, labels)
        top_family, _ = attr.attribute(0.9).top()
        assert top_family in ("gan", "diffusion", "other")

    def test_save_load_roundtrip_heuristic(self, tmp_path) -> None:
        attr = FamilyAttributor()
        path = tmp_path / "attributor.joblib"
        attr.save(path)
        loaded = FamilyAttributor.load(path)
        assert loaded.mode == "heuristic"
        assert loaded.attribute(0.5).probabilities == attr.attribute(0.5).probabilities

    def test_save_load_roundtrip_supervised(self, tmp_path) -> None:
        attr = FamilyAttributor()
        rng = np.random.default_rng(13)
        oof = rng.uniform(0.0, 1.0, 60)
        labels = rng.integers(0, len(FAMILIES), 60)
        labels[: len(FAMILIES)] = np.arange(len(FAMILIES))
        attr.fit_supervised(oof, labels)
        path = tmp_path / "attributor_sup.joblib"
        attr.save(path)
        loaded = FamilyAttributor.load(path)
        assert loaded.mode == "supervised"
        assert loaded.attribute(0.7).probabilities == attr.attribute(0.7).probabilities
