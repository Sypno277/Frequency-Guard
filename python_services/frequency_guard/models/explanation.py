"""Linguistic state collapse — NLP synthesis (FrequencyGuard Phase 4).

Turns the structured JSON from Phase 1 (provenance), Phase 2 (forensic /
ML), and Phase 3 (semantic arbiter) into human-readable, evidence-grounded
narrative with a severity prefix::

    [DETERMINISTIC AI] High confidence of synthetic generation. Cryptographic
    provenance is absent (Provenance Void). Artifact signature is most
    consistent with latent diffusion (62.1%). Ensemble agreement is high;
    strongest members: rf 100%, svm 99%.

This is deterministic and templated — no LLM, no API key, no GPU. It is
grounded in the *actual* measured signals, and the phrasing is
confidence-gated so a low-agreement / uncertain call never reads as a
confident claim.

Severity levels:
  - ``[DETERMINISTIC AI]``      confident AI verdict (state == "ai")
  - ``[PROBABLE AI]``           low-agreement AI or semantic flags present
  - ``[INCONCLUSIVE]``          verdict_state == "uncertain"
  - ``[DETERMINISTIC REAL]``    confident authentic verdict
  - ``[PROVENANCE VERIFIED]``   a C2PA manifest short-circuited the pipeline
"""

from __future__ import annotations

from dataclasses import dataclass

SEVERITY_DETERMINISTIC_AI = "[DETERMINISTIC AI]"
SEVERITY_PROBABLE_AI = "[PROBABLE AI]"
SEVERITY_INCONCLUSIVE = "[INCONCLUSIVE]"
SEVERITY_DETERMINISTIC_REAL = "[DETERMINISTIC REAL]"
SEVERITY_PROVENANCE_VERIFIED = "[PROVENANCE VERIFIED]"

_GENERATOR_LABELS: dict[str, str] = {
    "real": "real camera",
    "diffusion": "latent diffusion",
    "gan": "GAN",
    "other": "an unknown/other generator",
}


@dataclass
class ExplanationResult:
    """The natural-language verdict summary.

    ``severity`` is the bracketed prefix; ``narrative`` is the body.
    """

    severity: str
    narrative: str
    sentence_count: int = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "severity": self.severity,
            "narrative": self.narrative,
            "sentence_count": self.sentence_count,
        }


def _severity_for(
    verdict_state: str,
    agreement: float,
    semantic_suspicious: bool,
    provenance_bifurcated: bool,
    provenance_void: bool,
) -> str:
    """Pick the bracketed severity prefix from the combined signals."""
    if provenance_bifurcated:
        return SEVERITY_PROVENANCE_VERIFIED
    if verdict_state == "ai":
        # A confident AI is deterministic only if agreement is high.
        return SEVERITY_DETERMINISTIC_AI if agreement >= 0.5 else SEVERITY_PROBABLE_AI
    if verdict_state == "uncertain":
        # Semantic flags push a low-confidence call toward "probable AI".
        return SEVERITY_PROBABLE_AI if semantic_suspicious else SEVERITY_INCONCLUSIVE
    if verdict_state == "authentic":
        return SEVERITY_DETERMINISTIC_REAL
    return SEVERITY_INCONCLUSIVE


def _build_narrative(
    verdict_state: str,
    confidence: float,
    fake_probability: float,
    provenance_summary: str,
    provenance_void: bool,
    semantic_flags: list[object],
    top_family: str,
    top_family_prob: float,
    agreement: float,
    member_probabilities: dict[str, float],
    top_contributions: list[tuple[str, float]],
    generator_attribution: dict[str, float] | None = None,
) -> str:
    """Compose grounded, templated sentences from the measured signals."""
    sentences: list[str] = []

    # 1) Lead verdict sentence.
    if verdict_state == "ai":
        sentences.append(
            f"High confidence of synthetic generation ({confidence:.1f}% confidence)."
        )
    elif verdict_state == "authentic":
        sentences.append(
            f"Strong indication this is an authentic camera image "
            f"({confidence:.1f}% confidence in real)."
        )
    else:
        sentences.append(
            f"Verdict is inconclusive ({fake_probability * 100:.1f}% fake probability)."
        )

    # 2) Provenance sentence.
    if provenance_void:
        sentences.append(
            "Cryptographic provenance is absent (Provenance Void) — metadata was "
            "stripped, a common adversarial or social re-upload tactic."
        )
    elif provenance_summary and provenance_summary != "No metadata detected.":
        sentences.append(provenance_summary.rstrip(".") + ".")

    # 3) Signal-source / attribution sentence (family or per-generator).
    if generator_attribution:
        top_gens = sorted(generator_attribution.items(), key=lambda kv: kv[1], reverse=True)[:2]
        parts = ", ".join(f"{g} {p * 100:.1f}%" for g, p in top_gens)
        sentences.append(f"Generator attribution: {parts}.")
    else:
        family_label = _GENERATOR_LABELS.get(top_family, top_family)
        if top_family_prob >= 0.3:
            sentences.append(
                f"Artifact signature is most consistent with {family_label} "
                f"({top_family_prob * 100:.1f}%)."
            )

    # 4) Ensemble-agreement sentence (grounded in member probabilities).
    if agreement < 0.5:
        sentences.append(
            "Ensemble members disagree substantially, so the hard verdict is downgraded."
        )
    elif member_probabilities:
        top_members = sorted(member_probabilities.items(), key=lambda kv: kv[1], reverse=True)[:2]
        joined = ", ".join(f"{name} {prob * 100:.0f}%" for name, prob in top_members)
        sentences.append(f"Ensemble agreement is high; strongest members: {joined}.")

    # 5) Semantic-flag sentences (only present when the arbiter ran).
    for flag_obj in semantic_flags[:3]:
        try:
            label = str(flag_obj.get("label", ""))  # type: ignore[union-attr]
            score = float(flag_obj.get("score", 0.0))  # type: ignore[union-attr]
        except (AttributeError, ValueError, TypeError):
            continue
        if label and score >= 0.5:
            sentences.append(f"Semantic arbitration flagged: {label.lower()} (severity {score:.2f}).")

    # 6) Top feature contributions (grounded explainability).
    if top_contributions:
        feat, contrib = top_contributions[0]
        direction = "pushing toward fake" if contrib > 0 else "pushing toward real"
        sentences.append(
            f"Strongest single signal is {feat} ({abs(contrib):.2f}), {direction}."
        )

    return " ".join(sentences)


def synthesize_explanation(
    verdict_state: str,
    confidence: float,
    fake_probability: float,
    agreement: float,
    provenance_summary: str = "No metadata detected.",
    provenance_void: bool = False,
    provenance_bifurcated: bool = False,
    semantic_flags: list[object] | None = None,
    top_family: str = "other",
    top_family_prob: float = 0.0,
    member_probabilities: dict[str, float] | None = None,
    top_contributions: list[tuple[str, float]] | None = None,
    generator_attribution: dict[str, float] | None = None,
) -> ExplanationResult:
    """Fuse all signals into a severity-tagged natural-language summary.

    Args:
        verdict_state: "ai" | "authentic" | "uncertain".
        confidence: calibrated confidence (0..100).
        fake_probability: calibrated fake probability (0..1).
        agreement: ensemble agreement (1 = unanimous, 0 = split).
        provenance_summary: string from :func:`extract_provenance`.
        provenance_void: whether metadata was stripped.
        provenance_bifurcated: whether a C2PA manifest short-circuited.
        semantic_flags: list of dicts from the semantic arbiter.
        top_family: most likely generator family.
        top_family_prob: probability of that family.
        member_probabilities: {member name: fake probability}.
        top_contributions: [(feature, contribution)] sorted by |contribution|.
    """
    flags = semantic_flags or []
    members = member_probabilities or {}
    contribs = top_contributions or []
    semantic_suspicious = any(
        (isinstance(f, dict) and float(f.get("score", 0.0)) >= 0.5) for f in flags
    )

    severity = _severity_for(
        verdict_state,
        agreement,
        semantic_suspicious,
        provenance_bifurcated,
        provenance_void,
    )
    narrative = _build_narrative(
        verdict_state,
        confidence,
        fake_probability,
        provenance_summary,
        provenance_void,
        flags,
        top_family,
        top_family_prob,
        agreement,
        members,
        contribs,
        generator_attribution,
    )
    sentence_count = narrative.count(". ") + (1 if narrative.endswith(".") else 0)
    return ExplanationResult(severity=severity, narrative=narrative, sentence_count=sentence_count)
