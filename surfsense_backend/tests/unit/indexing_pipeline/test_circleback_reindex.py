"""Circleback evidence regression tests for the indexing pipeline."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest

from app.db import DocumentType
from app.indexing_pipeline import indexing_pipeline_service
from app.indexing_pipeline.cache.cached_indexing import EmbeddedChunk
from app.indexing_pipeline.document_chunker import chunk_text_hybrid

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def test_circleback_scratch_reindex_keeps_speaker_on_every_chunk(
    monkeypatch,
):
    speech = " ".join(
        f"Sentence {i}: production deployment needs review." for i in range(1, 100)
    )

    markdown = (
        "# Engineering Meeting\n\n"
        "## Transcript\n\n"
        f"**Alice** [01:00:00]: {speech}\n\n"
        "**Bob** [05:20]: I agree with the release plan.\n"
    )

    async def fake_generic_embeddings(content, *, use_code_chunker):
        texts = chunk_text_hybrid(content)

        return (
            np.zeros(384, dtype=np.float32),
            [
                EmbeddedChunk(
                    text=text,
                    start_line=1,
                    end_line=1,
                    embedding=np.zeros(384, dtype=np.float32),
                )
                for text in texts
            ],
        )

    monkeypatch.setattr(
        indexing_pipeline_service,
        "build_chunk_embeddings",
        fake_generic_embeddings,
    )

    async def fake_embed_batch(texts):
        return [np.zeros(384, dtype=np.float32) for _ in texts]

    monkeypatch.setattr(
        indexing_pipeline_service,
        "embed_batch",
        fake_embed_batch,
    )

    session = SimpleNamespace(execute=AsyncMock())
    service = indexing_pipeline_service.IndexingPipelineService(session)

    document = SimpleNamespace(
        id=777,
        document_type=DocumentType.CIRCLEBACK,
        embedding=None,
    )
    connector_doc = SimpleNamespace(should_use_code_chunker=False)

    chunks = await service._reindex_from_scratch(
        document,
        markdown,
        connector_doc,
    )

    alice_chunks = [chunk.content for chunk in chunks if "Sentence " in chunk.content]

    assert len(alice_chunks) > 1
    assert all(text.startswith("**Alice** [01:00:00]:") for text in alice_chunks), (
        "Scratch reindex must preserve Circleback evidence attribution."
    )


async def test_circleback_incremental_reindex_preserves_attribution(
    monkeypatch,
):
    from unittest.mock import MagicMock

    from app.indexing_pipeline.chunk_reconciler import ExistingChunk

    speech = " ".join(
        f"Sentence {i}: verify deployment evidence." for i in range(1, 100)
    )

    markdown = (
        "# Weekly Meeting\n\n"
        "## Transcript\n\n"
        f"**Alice** [01:00:00]: {speech}\n\n"
        "**Bob** [03:20]: I will review the results.\n"
    )

    async def fake_embed_batch(texts):
        return [np.zeros(384, dtype=np.float32) for _ in texts]

    monkeypatch.setattr(
        indexing_pipeline_service,
        "embed_batch",
        fake_embed_batch,
    )

    session = SimpleNamespace(
        execute=AsyncMock(),
        add_all=MagicMock(),
    )

    service = indexing_pipeline_service.IndexingPipelineService(session)

    document = SimpleNamespace(
        id=778,
        document_type=DocumentType.CIRCLEBACK,
        embedding=None,
    )

    connector_doc = SimpleNamespace(should_use_code_chunker=False)

    existing = [
        ExistingChunk(
            id=1,
            content="OLD_MEETING_EVIDENCE",
            position=0,
            start_line=1,
            end_line=1,
        )
    ]

    count = await service._reindex_incrementally(
        document,
        markdown,
        connector_doc,
        existing,
    )

    session.add_all.assert_called_once()

    inserted = list(session.add_all.call_args.args[0])
    alice_chunks = [chunk.content for chunk in inserted if "Sentence " in chunk.content]

    assert count >= len(inserted)
    assert len(alice_chunks) > 1
    assert all(text.startswith("**Alice** [01:00:00]:") for text in alice_chunks), (
        "Incremental reindex lost the speaker or timestamp."
    )

    assert any(chunk.content.startswith("**Bob** [03:20]:") for chunk in inserted)

    assert all("OLD_MEETING_EVIDENCE" not in chunk.content for chunk in inserted)
