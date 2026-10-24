"""Positive and negative Circleback webhook authentication cases."""

import hashlib
import hmac
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.db import get_async_session
from app.routes.circleback_webhook_auth import (
    verify_circleback_signature,
)
from app.routes.circleback_webhook_route import router
from app.tasks.celery_tasks.document_tasks import (
    process_circleback_meeting_task,
)

pytestmark = pytest.mark.unit

SECRETS = {
    "123": "workspace-123-test-secret",
    "456": "workspace-456-test-secret",
}


def sign(secret: str, body: bytes) -> str:
    return hmac.new(
        secret.encode("utf-8"),
        body,
        hashlib.sha256,
    ).hexdigest()


@pytest.fixture
def configured_secrets(monkeypatch):
    monkeypatch.setenv(
        "MEETINGOPS_CIRCLEBACK_SIGNING_SECRETS",
        json.dumps(SECRETS),
    )


def test_valid_signature_is_accepted(configured_secrets):
    body = b'{"meeting":101}'

    assert verify_circleback_signature(123, body, sign(SECRETS["123"], body))


def test_tampered_body_is_rejected(configured_secrets):
    original = b'{"meeting":101}'
    tampered = b'{"meeting":102}'

    assert not verify_circleback_signature(
        123,
        tampered,
        sign(SECRETS["123"], original),
    )


def test_workspace_secrets_cannot_be_reused(configured_secrets):
    body = b'{"meeting":101}'

    assert not verify_circleback_signature(
        456,
        body,
        sign(SECRETS["123"], body),
    )


@pytest.mark.parametrize(
    "signature",
    [None, "", "invalid", "0" * 64],
)
def test_invalid_signatures_are_rejected(configured_secrets, signature):
    assert not verify_circleback_signature(123, b'{"meeting":101}', signature)


def test_missing_secret_configuration_fails_closed(monkeypatch):
    monkeypatch.delenv(
        "MEETINGOPS_CIRCLEBACK_SIGNING_SECRETS",
        raising=False,
    )

    body = b'{"meeting":101}'

    assert not verify_circleback_signature(123, body, sign(SECRETS["123"], body))


def test_valid_signed_http_request_queues_meeting(
    monkeypatch,
    configured_secrets,
):
    queued = Mock()

    monkeypatch.setattr(
        process_circleback_meeting_task,
        "delay",
        queued,
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

    # Preserve the original bytes that Circleback would sign.
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    with TestClient(app) as client:
        response = client.post(
            "/webhooks/circleback/123",
            content=body,
            headers={
                "Content-Type": "application/json",
                "x-signature": sign(SECRETS["123"], body),
            },
        )

    assert response.status_code == 200, response.text
    assert response.json()["status"] == "accepted"

    fake_session.execute.assert_awaited_once()
    queued.assert_called_once()


def test_internal_processing_error_does_not_leak_details(
    monkeypatch,
    configured_secrets,
):
    """Internal exception details must never reach webhook callers."""

    sensitive_marker = "PRIVATE_INTERNAL_CREDENTIAL_MARKER"

    monkeypatch.setattr(
        process_circleback_meeting_task,
        "delay",
        Mock(side_effect=RuntimeError(sensitive_marker)),
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
        "id": 102,
        "name": "Security Review",
        "createdAt": "2026-10-24T10:00:00Z",
        "duration": 600,
    }

    body = json.dumps(payload).encode("utf-8")

    with TestClient(app) as client:
        response = client.post(
            "/webhooks/circleback/123",
            content=body,
            headers={
                "Content-Type": "application/json",
                "x-signature": sign(SECRETS["123"], body),
            },
        )

    assert response.status_code == 500

    assert sensitive_marker not in response.text, (
        "Internal exception details leaked to the webhook caller."
    )


def test_duplicate_signed_delivery_queues_only_once(
    monkeypatch,
    configured_secrets,
):
    """An identical signed delivery must not enqueue duplicate work."""
    queued = Mock()

    monkeypatch.setattr(
        process_circleback_meeting_task,
        "delay",
        queued,
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
    }

    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "x-signature": sign(SECRETS["123"], body),
    }

    with TestClient(app) as client:
        first = client.post(
            "/webhooks/circleback/123",
            content=body,
            headers=headers,
        )
        second = client.post(
            "/webhooks/circleback/123",
            content=body,
            headers=headers,
        )

    assert first.status_code == 200
    assert second.status_code == 200

    assert queued.call_count == 1, (
        "Duplicate webhook delivery queued duplicate Celery work."
    )


@pytest.fixture(autouse=True)
def fake_delivery_redis(monkeypatch):
    """In-memory Redis test double, shared within each test case."""
    from app.routes import circleback_webhook_idempotency

    class FakeRedis:
        def __init__(self):
            self.values = {}

        def set(self, key, value, *, nx=False, ex=None):
            if nx and key in self.values:
                return None
            self.values[key] = value
            return True

        def eval(self, script, numkeys, key, token):
            if self.values.get(key) == token:
                del self.values[key]
                return 1
            return 0

    client = FakeRedis()

    monkeypatch.setattr(
        circleback_webhook_idempotency,
        "_client",
        lambda: client,
    )

    return client


def _build_signed_webhook_test_client():
    """Build a lightweight HTTP client with a mocked database."""
    fake_session = SimpleNamespace(
        execute=AsyncMock(return_value=SimpleNamespace(scalar_one_or_none=lambda: 456))
    )

    async def override_session():
        yield fake_session

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_async_session] = override_session
    return TestClient(app)


def _post_signed(client, meeting_id, name="Release Planning"):
    body = json.dumps(
        {
            "id": meeting_id,
            "name": name,
            "createdAt": "2026-10-24T10:00:00Z",
            "duration": 3600,
        }
    ).encode("utf-8")

    return client.post(
        "/webhooks/circleback/123",
        content=body,
        headers={
            "Content-Type": "application/json",
            "x-signature": sign(SECRETS["123"], body),
        },
    )


def test_redis_failure_rejects_delivery_without_queueing(
    monkeypatch,
    configured_secrets,
):
    """Redis outage must not silently bypass deduplication."""
    import redis

    from app.routes import circleback_webhook_idempotency

    class BrokenRedis:
        def set(self, *args, **kwargs):
            raise redis.exceptions.ConnectionError("Redis unavailable")

    monkeypatch.setattr(
        circleback_webhook_idempotency,
        "_client",
        lambda: BrokenRedis(),
    )

    queued = Mock()
    monkeypatch.setattr(process_circleback_meeting_task, "delay", queued)

    with _build_signed_webhook_test_client() as client:
        response = _post_signed(client, 201)

    assert response.status_code == 503
    queued.assert_not_called()


def test_failed_enqueue_releases_claim_for_retry(
    monkeypatch,
    configured_secrets,
    fake_delivery_redis,
):
    """A definite enqueue exception must permit another attempt."""
    queued = Mock(
        side_effect=[
            RuntimeError("broker unavailable"),
            None,
        ]
    )
    monkeypatch.setattr(process_circleback_meeting_task, "delay", queued)

    with _build_signed_webhook_test_client() as client:
        first = _post_signed(client, 202)
        assert first.status_code == 500
        assert fake_delivery_redis.values == {}

        second = _post_signed(client, 202)

    assert second.status_code == 200
    assert second.json()["status"] == "accepted"
    assert queued.call_count == 2


def test_updated_meeting_is_not_treated_as_duplicate(
    monkeypatch,
    configured_secrets,
):
    """A genuine content change must be accepted for processing."""
    queued = Mock()
    monkeypatch.setattr(process_circleback_meeting_task, "delay", queued)

    with _build_signed_webhook_test_client() as client:
        original = _post_signed(client, 203, name="Release Planning")
        updated = _post_signed(client, 203, name="Updated Release Planning")

    assert original.status_code == 200
    assert updated.status_code == 200
    assert queued.call_count == 2


def _build_signed_webhook_test_client():
    """Build a lightweight HTTP client with a mocked database."""
    fake_session = SimpleNamespace(
        execute=AsyncMock(return_value=SimpleNamespace(scalar_one_or_none=lambda: 456))
    )

    async def override_session():
        yield fake_session

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_async_session] = override_session
    return TestClient(app)


def _post_signed(client, meeting_id, name="Release Planning"):
    body = json.dumps(
        {
            "id": meeting_id,
            "name": name,
            "createdAt": "2026-10-24T10:00:00Z",
            "duration": 3600,
        }
    ).encode("utf-8")

    return client.post(
        "/webhooks/circleback/123",
        content=body,
        headers={
            "Content-Type": "application/json",
            "x-signature": sign(SECRETS["123"], body),
        },
    )


def test_signed_string_meeting_id_reaches_celery(
    monkeypatch,
    configured_secrets,
):
    """Official string meeting IDs reach Celery unchanged."""
    meeting_id = "WRweb12iCBimU6Vo8cD7z"

    queued = Mock()
    monkeypatch.setattr(
        process_circleback_meeting_task,
        "delay",
        queued,
    )

    with _build_signed_webhook_test_client() as client:
        response = _post_signed(
            client,
            meeting_id,
            name="Event Venue Review",
        )

    assert response.status_code == 200, response.text
    assert response.json()["meeting_id"] == meeting_id

    queued.assert_called_once()
    assert queued.call_args.kwargs["meeting_id"] == meeting_id
    assert (
        queued.call_args.kwargs["metadata"]["circleback_meeting_id"]
        == meeting_id
    )
