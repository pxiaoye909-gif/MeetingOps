"""Tests for Circleback transcript formatting."""

import pytest

from app.routes.circleback_webhook_route import (
    CirclebackWebhookPayload,
    format_circleback_meeting_to_markdown,
)

pytestmark = pytest.mark.unit


def _meeting_markdown(timestamp: float) -> str:
    payload = CirclebackWebhookPayload(
        id=101,
        name="Release Planning",
        createdAt="2026-10-22T10:00:00Z",
        duration=3600.0,
        transcript=[
            {
                "speaker": "Alice",
                "text": "We should delay the release.",
                "timestamp": timestamp,
            }
        ],
    )
    return format_circleback_meeting_to_markdown(payload)


def test_transcript_preserves_speaker_timestamp_and_text() -> None:
    markdown = _meeting_markdown(83.0)

    assert "**Alice** [01:23]: We should delay the release." in markdown


def test_long_meeting_timestamp_includes_hours() -> None:
    markdown = _meeting_markdown(3600.0)

    assert "**Alice** [01:00:00]: We should delay the release." in markdown


def test_circleback_accepts_official_string_meeting_id() -> None:
    """Circleback documents string meeting IDs."""
    meeting_id = "WRweb12iCBimU6Vo8cD7z"

    payload = CirclebackWebhookPayload(
        id=meeting_id,
        name="Event Venue Review",
        createdAt="2026-10-24T10:00:00Z",
        duration=600,
    )

    assert payload.id == meeting_id
    assert meeting_id in format_circleback_meeting_to_markdown(payload)
