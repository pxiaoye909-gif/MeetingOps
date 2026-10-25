"""Circleback Git identity metadata contract."""

import json

import pytest

from app.knowledge_store.circleback_identity import (
    decode_identity,
    encode_identity,
    identity_path,
)

pytestmark = pytest.mark.unit

PATH = "/documents/Release Review.md"
BODY = "# Release Review\n\n**Alice** [00:03]: Approved.\n"
MEETING_ID = "WRweb12iCBimU6Vo8cD7z"


def test_identity_roundtrip():
    encoded = encode_identity(
        virtual_path=PATH,
        meeting_id=MEETING_ID,
        markdown=BODY,
    )
    assert decode_identity(encoded, virtual_path=PATH, markdown=BODY) == {
        "CIRCLEBACK_MEETING_ID": MEETING_ID,
        "SOURCE": "CIRCLEBACK_WEBHOOK",
    }


def test_identity_rejects_changed_content():
    encoded = encode_identity(
        virtual_path=PATH,
        meeting_id=MEETING_ID,
        markdown=BODY,
    )
    assert (
        decode_identity(
            encoded,
            virtual_path=PATH,
            markdown=BODY + "Changed",
        )
        is None
    )


def test_identity_rejects_different_path():
    encoded = encode_identity(
        virtual_path=PATH,
        meeting_id=MEETING_ID,
        markdown=BODY,
    )
    assert (
        decode_identity(
            encoded,
            virtual_path="/documents/Another.md",
            markdown=BODY,
        )
        is None
    )


def test_identity_rejects_wrong_type():
    encoded = json.loads(
        encode_identity(
            virtual_path=PATH,
            meeting_id=MEETING_ID,
            markdown=BODY,
        )
    )
    encoded["document_type"] = "NOTE"
    assert (
        decode_identity(
            json.dumps(encoded),
            virtual_path=PATH,
            markdown=BODY,
        )
        is None
    )


def test_identity_rejects_malformed_data():
    assert (
        decode_identity(
            "{bad json",
            virtual_path=PATH,
            markdown=BODY,
        )
        is None
    )


def test_identity_path_is_outside_documents():
    result = identity_path(PATH)
    assert result.startswith("meetingops/identities/")
    assert result.endswith(".json")
