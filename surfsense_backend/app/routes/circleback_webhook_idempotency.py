"""Bounded Redis delivery deduplication for Circleback webhooks."""

import hashlib
import secrets
from functools import lru_cache

import redis

from app.config import config

DELIVERY_WINDOW_SECONDS = 300

_RELEASE_IF_OWNER = """
if redis.call("GET", KEYS[1]) == ARGV[1] then
    return redis.call("DEL", KEYS[1])
end
return 0
"""


class DeliveryStoreUnavailableError(Exception):
    """The shared deduplication store cannot be used."""


@lru_cache(maxsize=1)
def _client():
    return redis.Redis.from_url(
        config.REDIS_APP_URL,
        decode_responses=True,
        socket_connect_timeout=2,
        socket_timeout=2,
    )


def _key(workspace_id: int, raw_body: bytes) -> str:
    digest = hashlib.sha256(raw_body).hexdigest()
    return f"meetingops:circleback:delivery:v1:{workspace_id}:{digest}"


def claim_circleback_delivery(
    workspace_id: int,
    raw_body: bytes,
) -> str | None:
    """Atomically claim an unseen delivery for a bounded time window.

    Returns an ownership token, or None for a duplicate.
    """
    token = secrets.token_hex(24)

    try:
        acquired = _client().set(
            _key(workspace_id, raw_body),
            token,
            nx=True,
            ex=DELIVERY_WINDOW_SECONDS,
        )
    except (redis.exceptions.RedisError, OSError, ValueError) as exc:
        raise DeliveryStoreUnavailableError from exc

    return token if acquired else None


def release_circleback_delivery(
    workspace_id: int,
    raw_body: bytes,
    token: str,
) -> None:
    """Release a failed enqueue claim only if we still own it."""
    try:
        _client().eval(
            _RELEASE_IF_OWNER,
            1,
            _key(workspace_id, raw_body),
            token,
        )
    except (redis.exceptions.RedisError, OSError, ValueError) as exc:
        raise DeliveryStoreUnavailableError from exc
