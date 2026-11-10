"""File fingerprints that do not depend on line endings.

Git on Windows checks text files out with CRLF line endings and on the Mac
with LF, so the same file used to get two different SHA-256 digests, and every
report flagged "calibration changed" between machines (AUDIT_2026-11-06.md M2).

From 2026-11-09 a file's digest is taken over its bytes with every CRLF turned
into LF. On the Mac, where files are already LF, that is the same digest as
before, so the Mac runs' manifests stay valid. Manifests written on Windows
before that date hold the CRLF digest; `matches` accepts either form, so
those runs can still be checked against today's files.
"""

import hashlib
from pathlib import Path


def lf_bytes(data: bytes) -> bytes:
    return data.replace(b"\r\n", b"\n")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    """Digest of the file with line endings normalized to LF."""
    return sha256_bytes(lf_bytes(Path(path).read_bytes()))


def digest_variants(data: bytes) -> dict[str, str]:
    """The digests the same text has with LF and with CRLF line endings,
    mapped to the LF (canonical) digest."""
    lf = lf_bytes(data)
    canonical = sha256_bytes(lf)
    return {canonical: canonical, sha256_bytes(lf.replace(b"\n", b"\r\n")): canonical}


def matches(path: Path, digest: str) -> bool:
    """True when `digest` is this file's digest with either line ending."""
    return digest in digest_variants(Path(path).read_bytes())
