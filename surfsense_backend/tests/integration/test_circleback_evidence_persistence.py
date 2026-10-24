"""PostgreSQL integration test for attributed Circleback evidence."""

import pytest
from sqlalchemy import select

from app.config import config
from app.db import Chunk, Document, DocumentType
from app.tasks.document_processors import (
    circleback_chunking,
    circleback_processor,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


async def test_circleback_evidence_survives_database_roundtrip(
    db_session,
    db_workspace,
    monkeypatch,
    workspace_flip,
):
    # Isolate this test from external knowledge-store writes.
    workspace_flip(False)

    dimension = config.embedding_model_instance.dimension
    embedded_texts = []

    monkeypatch.setattr(
        circleback_processor,
        "embed_text",
        lambda text: [0.1] * dimension,
    )

    def fake_embed_texts(texts):
        embedded_texts.extend(texts)
        return [[0.2] * dimension for _ in texts]

    monkeypatch.setattr(
        circleback_chunking,
        "embed_texts",
        fake_embed_texts,
    )

    long_speech = " ".join(
        f"Sentence {i}: review the release blockers." for i in range(100)
    )

    markdown = (
        "# Release Planning\n\n"
        "## Meeting Notes\n\n"
        "Decision: postpone the release.\n\n"
        "## Transcript\n\n"
        f"**Alice** [01:00:00]: {long_speech}\n\n"
        "**Bob** [02:15]: I will update the deployment plan.\n"
    )

    document = await circleback_processor.add_circleback_meeting_document(
        session=db_session,
        meeting_id=987654,
        meeting_name="Release Planning",
        markdown_content=markdown,
        metadata={},
        workspace_id=db_workspace.id,
        connector_id=None,
    )

    assert document is not None
    document_id = document.id

    # Remove ORM-cached objects so the following queries read from PostgreSQL.
    db_session.expunge_all()

    saved_document = await db_session.get(Document, document_id)

    assert saved_document is not None
    assert saved_document.document_type == DocumentType.CIRCLEBACK
    assert saved_document.source_markdown == markdown

    result = await db_session.execute(
        select(Chunk).where(Chunk.document_id == document_id).order_by(Chunk.position)
    )
    saved_chunks = result.scalars().all()

    assert len(saved_chunks) > 2
    assert [chunk.position for chunk in saved_chunks] == list(range(len(saved_chunks)))

    stored_texts = [chunk.content for chunk in saved_chunks]

    # The text embedded must be exactly the text stored in PostgreSQL.
    assert stored_texts == embedded_texts

    alice_chunks = [chunk for chunk in saved_chunks if "Sentence " in chunk.content]

    assert len(alice_chunks) > 1
    assert all(
        chunk.content.startswith("**Alice** [01:00:00]:") for chunk in alice_chunks
    )

    assert any(
        chunk.content.startswith("**Bob** [02:15]:")
        and "deployment plan" in chunk.content
        for chunk in saved_chunks
    )

    assert all(
        chunk.embedding is not None
        and len(chunk.embedding) == dimension
        and chunk.embedding[0] == pytest.approx(0.2)
        for chunk in saved_chunks
    )


async def test_circleback_update_replaces_stale_chunks(
    db_session,
    db_workspace,
    monkeypatch,
    workspace_flip,
):
    """Updating a meeting must remove obsolete evidence."""
    workspace_flip(False)

    dimension = config.embedding_model_instance.dimension

    monkeypatch.setattr(
        circleback_processor,
        "embed_text",
        lambda text: [0.1] * dimension,
    )
    monkeypatch.setattr(
        circleback_chunking,
        "embed_texts",
        lambda texts: [[0.2] * dimension for _ in texts],
    )

    original = (
        "# Release Meeting\n\n"
        "## Transcript\n\n"
        "**Alice** [01:00]: OLD_DECISION_FRIDAY_RELEASE.\n"
    )
    updated = (
        "# Release Meeting\n\n"
        "## Transcript\n\n"
        "**Alice** [01:00]: NEW_DECISION_MONDAY_RELEASE.\n"
    )

    async def save(markdown):
        return await circleback_processor.add_circleback_meeting_document(
            session=db_session,
            meeting_id="WRweb12iCBimU6Vo8cD7z",
            meeting_name="Release Meeting",
            markdown_content=markdown,
            metadata={},
            workspace_id=db_workspace.id,
            connector_id=None,
        )

    first = await save(original)
    assert first is not None

    document_id = first.id

    # Same Circleback ID and unchanged content must reuse the document.
    unchanged = await save(original)
    assert unchanged is not None
    assert unchanged.id == document_id

    old_rows = (
        (
            await db_session.execute(
                select(Chunk).where(Chunk.document_id == document_id)
            )
        )
        .scalars()
        .all()
    )

    old_ids = {chunk.id for chunk in old_rows}
    assert old_ids
    assert any("OLD_DECISION_FRIDAY_RELEASE" in chunk.content for chunk in old_rows)

    second = await save(updated)
    assert second is not None
    assert second.id == document_id

    db_session.expunge_all()

    saved_document = await db_session.get(Document, document_id)
    assert saved_document.source_markdown == updated
    assert (
        saved_document.document_metadata["CIRCLEBACK_MEETING_ID"]
        == "WRweb12iCBimU6Vo8cD7z"
    )

    matches = (
        await db_session.execute(
            select(Document).where(
                Document.unique_identifier_hash
                == saved_document.unique_identifier_hash
            )
        )
    ).scalars().all()
    assert len(matches) == 1

    new_rows = (
        (
            await db_session.execute(
                select(Chunk)
                .where(Chunk.document_id == document_id)
                .order_by(Chunk.position)
            )
        )
        .scalars()
        .all()
    )

    assert new_rows
    assert not old_ids.intersection({chunk.id for chunk in new_rows})

    new_texts = [chunk.content for chunk in new_rows]

    assert any("NEW_DECISION_MONDAY_RELEASE" in text for text in new_texts)
    assert all("OLD_DECISION_FRIDAY_RELEASE" not in text for text in new_texts)
    assert any(text.startswith("**Alice** [01:00]:") for text in new_texts)


@pytest.mark.parametrize(
    "incremental",
    [False, True],
    ids=["scratch", "incremental"],
)
async def test_circleback_reindex_preserves_database_evidence(
    db_session,
    db_workspace,
    monkeypatch,
    workspace_flip,
    incremental,
):
    import numpy as np

    from app.db import DocumentStatus
    from app.indexing_pipeline import indexing_pipeline_service
    from app.indexing_pipeline.connector_document import ConnectorDocument

    workspace_flip(False)

    dimension = config.embedding_model_instance.dimension
    vector = np.zeros(dimension, dtype=np.float32)

    monkeypatch.setattr(
        circleback_processor,
        "embed_text",
        lambda text: vector,
    )
    monkeypatch.setattr(
        circleback_chunking,
        "embed_texts",
        lambda texts: [vector.copy() for _ in texts],
    )

    async def fake_embed_batch(texts):
        return [vector.copy() for _ in texts]

    monkeypatch.setattr(
        indexing_pipeline_service,
        "embed_batch",
        fake_embed_batch,
    )
    monkeypatch.setattr(
        config,
        "CHUNK_RECONCILE_ENABLED",
        incremental,
    )

    old_speech = " ".join(f"OLD_DECISION_{i} release on Friday." for i in range(100))
    new_speech = " ".join(f"NEW_DECISION_{i} release on Monday." for i in range(100))

    def make_markdown(speech):
        return (
            "# Release Meeting\n\n"
            "## Transcript\n\n"
            f"**Alice** [01:00:00]: {speech}\n\n"
            "**Bob** [02:30]: I will update the plan.\n"
        )

    original = make_markdown(old_speech)
    updated = make_markdown(new_speech)

    document = await circleback_processor.add_circleback_meeting_document(
        session=db_session,
        meeting_id=988100 if incremental else 988101,
        meeting_name="Release Meeting",
        markdown_content=original,
        metadata={},
        workspace_id=db_workspace.id,
    )

    assert document is not None
    document_id = document.id

    connector_doc = ConnectorDocument(
        title="Release Meeting",
        source_markdown=updated,
        unique_id="meeting-reindex-test",
        document_type=DocumentType.CIRCLEBACK,
        workspace_id=db_workspace.id,
        created_by_id=str(db_workspace.user_id),
    )

    # The upstream preparation stage updates source_markdown.
    document.source_markdown = updated

    service = indexing_pipeline_service.IndexingPipelineService(db_session)
    result = await service.index(document, connector_doc)

    assert DocumentStatus.is_state(
        result.status,
        DocumentStatus.READY,
    )

    await db_session.flush()
    db_session.expunge_all()

    rows = (
        (
            await db_session.execute(
                select(Chunk)
                .where(Chunk.document_id == document_id)
                .order_by(Chunk.position)
            )
        )
        .scalars()
        .all()
    )

    assert len(rows) > 2
    assert [row.position for row in rows] == list(range(len(rows)))

    texts = [row.content for row in rows]

    assert any("NEW_DECISION_" in text for text in texts)
    assert all("OLD_DECISION_" not in text for text in texts)

    assert all(
        text.startswith("**Alice** [01:00:00]:")
        for text in texts
        if "NEW_DECISION_" in text
    )

    assert any(text.startswith("**Bob** [02:30]:") for text in texts)

    assert all(
        row.embedding is not None
        and len(row.embedding) == dimension
        and row.start_line is not None
        and row.end_line is not None
        and row.start_line <= row.end_line
        for row in rows
    )
