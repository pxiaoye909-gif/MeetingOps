"""Versioned Circleback identity stored alongside Git document content."""

from __future__ import annotations

import hashlib
import json

SCHEMA_VERSION = 1
IDENTITY_ROOT = "meetingops/identities"


def identity_path(virtual_path: str) -> str:
    """Stable internal path; never interpreted as a user document."""
    if not virtual_path.startswith("/documents/"):
        raise ValueError("Invalid document path")
    digest = hashlib.sha256(virtual_path.encode("utf-8")).hexdigest()
    return f"{IDENTITY_ROOT}/{digest}.json"


def encode_identity(
    *,
    virtual_path: str,
    meeting_id: str,
    markdown: str,
) -> str:
    """Serialize the minimum identity needed to rebuild a meeting."""
    if not isinstance(meeting_id, str) or not meeting_id.strip():
        raise ValueError("Meeting ID must be a non-empty string")

    data = {
        "schema_version": SCHEMA_VERSION,
        "document_type": "CIRCLEBACK",
        "meeting_id": meeting_id,
        "source": "CIRCLEBACK_WEBHOOK",
        "virtual_path": virtual_path,
        "content_sha256": hashlib.sha256(markdown.encode("utf-8")).hexdigest(),
    }
    identity_path(virtual_path)
    return json.dumps(data, sort_keys=True, ensure_ascii=False)


def decode_identity(
    raw: bytes | str,
    *,
    virtual_path: str,
    markdown: str,
) -> dict[str, str] | None:
    """Reject missing, malformed, stale or mismatched identity records."""
    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            return None

        meeting_id = data.get("meeting_id")
        if not isinstance(meeting_id, str) or not meeting_id.strip():
            return None

        expected_hash = hashlib.sha256(markdown.encode("utf-8")).hexdigest()

        if (
            data.get("schema_version") != SCHEMA_VERSION
            or data.get("document_type") != "CIRCLEBACK"
            or data.get("source") != "CIRCLEBACK_WEBHOOK"
            or data.get("virtual_path") != virtual_path
            or data.get("content_sha256") != expected_hash
        ):
            return None

        return {
            "CIRCLEBACK_MEETING_ID": meeting_id,
            "SOURCE": "CIRCLEBACK_WEBHOOK",
        }
    except (ValueError, TypeError, UnicodeError):
        return None


async def read_identity(
    store,
    revision: str,
    *,
    virtual_path: str,
    markdown: str,
) -> dict[str, str] | None:
    """Read and validate Circleback identity at the same Git revision."""
    try:
        raw = await store.read_as_of(revision, identity_path(virtual_path))
    except (KeyError, FileNotFoundError):
        # Legacy Git document or ordinary NOTE without an identity sidecar.
        return None

    return decode_identity(
        raw,
        virtual_path=virtual_path,
        markdown=markdown,
    )
