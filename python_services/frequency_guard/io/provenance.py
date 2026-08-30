"""Provenance & cryptographic manifest extraction (FrequencyGuard Phase 1).

CPU-only. Reads the image's binary structure and surfaces metadata signals:
EXIF/IPTC tags, plus a best-effort C2PA / Content-Credentials manifest
detection. When a valid cryptographic manifest exists we bifurcate the
verdict; when metadata has been stripped, we mark a "Provenance Void" and
apply a probability penalty that routes the image to the ML/forensic
pipeline (Phase 2).

This module deliberately depends only on stdlib ``struct``/``bytes`` scans
so it runs in microseconds on any CPU, no external library required. It
cannot verify a C2PA signature cryptographically (that needs the c2pa
toolchain) — it *detects* the manifest block and reports it as a strong
attestation signal, and flags its own limitation honestly.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Any

# --- C2PA / ISO 21496 markers ------------------------------------------
# A Content-Credentials bundle is stored either in a JUMBF (JPEG Universal
# Metadata Box, Box-type "jumb") box, a PNG "caBX"/c2pa tEXt chunk, or a
# generic "c2pa" sidecar. We scan for the ASCII markers that precede a
# claim manifest so detection works across container types.

_C2PA_MARKERS = (b"jumb", b"c2pa", b"cai_", b"contentcredentials", b"claim_generator")
_EXIF_HEADER = b"Exif\x00\x00"
_IPTC_MARKERS = (b"Photoshop 3.0", b"iptc", b"XMLPacket")

# --- EXIF GPS / device tags we care about (TIFF tag IDs) ----------------
_EXIF_TAGS: dict[int, str] = {
    0x010F: "make",
    0x0110: "model",
    0x0132: "datetime",
    0x8827: "iso_speed",
    0x920A: "focal_length",
    0x829A: "exposure_time",
}


@dataclass
class ProvenanceResult:
    """Structured provenance signal for one image."""

    has_c2pa: bool = False
    has_exif: bool = False
    has_iptc: bool = False
    exif_tags: dict[str, Any] = field(default_factory=dict)
    provenance_void: bool = False  # True when the image was stripped of metadata
    void_penalty: float = 0.0  # 0..1 additive probability penalty
    summary: str = "No metadata detected."

    @property
    def bifurcated(self) -> bool:
        """True when a cryptographic manifest gives a deterministic verdict."""
        # A genuine C2PA signature is authoritative. We surface it as
        # bifurcated so the caller can short-circuit to a verdict without
        # running the ML pipeline. We cannot *cryptographically* verify it
        # here, so this is an attestation signal, not proof.
        return self.has_c2pa


def _find_marker(buf: bytes, marker: bytes) -> int:
    """Return byte offset of ``marker`` in ``buf``, or -1."""
    return buf.find(marker)


def _parse_exif_from_jpeg(buf: bytes) -> dict[str, Any]:
    """Best-effort EXIF tag extraction from a JPEG/JPEG-like buffer.

    Scans TIFF little/big-endian headers. No external Pillow dependency.
    Returns only tags in :data:`_EXIF_TAGS` that are present.
    """
    tags: dict[str, Any] = {}
    exif_off = buf.find(_EXIF_HEADER)
    if exif_off < 0 or exif_off + 8 > len(buf):
        return tags
    tiff = buf[exif_off + 6 : exif_off + 6 + 8]
    if len(tiff) < 8:
        return tags
    endian = tiff[:2]
    if endian == b"II":
        order = "<"
    elif endian == b"MM":
        order = ">"
    else:
        return tags
    try:
        magic = struct.unpack(order + "H", tiff[2:4])[0]
        if magic != 0x2A:
            return tags
        ifd_offset = struct.unpack(order + "I", tiff[4:8])[0]
    except struct.error:
        return tags

    base = exif_off + 6
    ifd_pos = base + ifd_offset
    try:
        n_entries = struct.unpack(order + "H", buf[ifd_pos : ifd_pos + 2])[0]
    except struct.error:
        return tags

    for i in range(n_entries):
        entry_pos = ifd_pos + 2 + i * 12
        if entry_pos + 12 > len(buf):
            break
        try:
            tag_id, tag_type, count = struct.unpack(order + "HHI", buf[entry_pos : entry_pos + 8])
        except struct.error:
            continue
        if tag_id not in _EXIF_TAGS:
            continue
        value_bytes = buf[entry_pos + 8 : entry_pos + 12]
        # ASCII (type 2) and rational/unsigned types are common. We decode
        # the 4-byte slot leniently; long strings need the offset branch.
        try:
            if tag_type == 2 and count <= 4:
                value = value_bytes.split(b"\x00", 1)[0].decode("utf-8", "replace")
            elif tag_type == 3 and count == 1:
                value = struct.unpack(order + "H", value_bytes[:2])[0]
            elif tag_type == 4 and count == 1:
                value = struct.unpack(order + "I", value_bytes)[0]
            else:
                value = value_bytes.hex()
        except (struct.error, UnicodeDecodeError):
            value = value_bytes.hex()
        tags[_EXIF_TAGS[tag_id]] = value
    return tags


def extract_provenance(buffer: bytes) -> ProvenanceResult:
    """Extract provenance/metadata signals from raw image bytes.

    Args:
        buffer: original uploaded file bytes.

    Returns:
        :class:`ProvenanceResult` with detected signals, the provenance-void
        flag, and an additive void penalty.
    """
    buf = buffer if isinstance(buffer, (bytes, bytearray)) else bytes(buffer)
    has_c2pa = any(_find_marker(buf, m) >= 0 for m in _C2PA_MARKERS)
    has_iptc = any(_find_marker(buf, m) >= 0 for m in _IPTC_MARKERS)
    exif_tags = _parse_exif_from_jpeg(buf)
    has_exif = bool(exif_tags)

    # Provenance Void: a JPEG/PNG that has *no* EXIF, IPTC, or C2PA is
    # suspicious because social platforms and adversarial re-saves strip
    # metadata. This is a soft, probabilistic penalty, not a verdict.
    # Only apply the void when the file is a known image container.
    is_image_container = (
        buf[:3] == b"\xff\xd8\xff"  # JPEG
        or buf[:8] == b"\x89PNG\r\n\x1a\n"  # PNG
        or buf[:2] == b"BM"  # BMP
        or buf[:4] == b"RIFF"  # WEBP
    )
    provenance_void = is_image_container and not (has_exif or has_iptc or has_c2pa)
    void_penalty = 0.10 if provenance_void else 0.0

    if has_c2pa:
        summary = "C2PA / Content-Credentials manifest detected — strong attestation signal."
    elif has_exif:
        summary = f"EXIF metadata detected ({len(exif_tags)} tags)."
    elif has_iptc:
        summary = "IPTC metadata detected."
    elif provenance_void:
        summary = "Provenance Void — metadata stripped (common adversarial/social re-upload)."
    else:
        summary = "No metadata detected."

    return ProvenanceResult(
        has_c2pa=has_c2pa,
        has_exif=has_exif,
        has_iptc=has_iptc,
        exif_tags=exif_tags,
        provenance_void=provenance_void,
        void_penalty=void_penalty,
        summary=summary,
    )
