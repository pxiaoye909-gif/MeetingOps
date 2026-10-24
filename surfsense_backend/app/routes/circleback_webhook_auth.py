"""Verify Circleback webhook signatures before processing events."""

import hashlib
import hmac
import json
import os


def verify_circleback_signature(
    workspace_id: int,
    raw_body: bytes,
    signature: str | None,
) -> bool:
    """Fail closed unless the workspace's HMAC-SHA256 matches."""
    if not signature:
        return False

    # Reject malformed signatures before constant-time comparison.
    if len(signature) != 64:
        return False

    try:
        raw_config = os.environ.get(
            "MEETINGOPS_CIRCLEBACK_SIGNING_SECRETS",
            "",
        )
        secrets = json.loads(raw_config)

        if not isinstance(secrets, dict):
            return False

        secret = secrets.get(str(workspace_id))

        if not isinstance(secret, str) or not secret:
            return False

        expected = hmac.new(
            secret.encode("utf-8"),
            raw_body,
            hashlib.sha256,
        ).hexdigest()

        return hmac.compare_digest(expected, signature)

    except (ValueError, TypeError, UnicodeError):
        return False
