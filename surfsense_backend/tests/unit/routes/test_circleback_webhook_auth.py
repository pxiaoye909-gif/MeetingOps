"""Security regression tests for Circleback webhook delivery."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.db import get_async_session
from app.routes.circleback_webhook_route import router
from app.tasks.celery_tasks.document_tasks import (
    process_circleback_meeting_task,
)

pytestmark = pytest.mark.unit


def test_unsigned_circleback_request_is_rejected(
    monkeypatch,
):
    # Only a dummy testing secret; never use real credentials here.
    monkeypatch.setenv(
        "MEETINGOPS_CIRCLEBACK_SIGNING_SECRETS",
        json.dumps({"123": "test-only-signing-secret"}),
    )

    queued_task = Mock()
    monkeypatch.setattr(
        process_circleback_meeting_task,
        "delay",
        queued_task,
    )

    fake_session = SimpleNamespace(
        execute=AsyncMock(return_value=SimpleNamespace(scalar_one_or_none=lambda: 456))
    )

    async def override_session():
        yield fake_session

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_async_session] = override_session

    payload = {
        "id": 101,
        "name": "Release Planning",
        "createdAt": "2026-10-24T10:00:00Z",
        "duration": 3600,
        "transcript": [
            {
                "speaker": "Alice",
                "text": "Review deployment security.",
                "timestamp": 65,
            }
        ],
    }

    with TestClient(app) as client:
        response = client.post(
            "/webhooks/circleback/123",
            json=payload,
        )

    assert response.status_code == 401, "Unsigned webhook requests must be rejected."
    queued_task.assert_not_called()
    fake_session.execute.assert_not_awaited()
