# ruff: noqa: RUF001
"""Tests for Circleback evidence-preserving chunking."""

import pytest

from app.tasks.document_processors.circleback_chunking import chunk_circleback_markdown

pytestmark = pytest.mark.unit


def test_long_utterance_preserves_speaker_and_timestamp():

    long_text = " ".join(
        f"Sentence {i}: release risks and deployment schedule." for i in range(1, 45)
    )

    markdown = (
        "# Release Planning\n\n"
        "## Transcript\n\n"
        f"**Alice** [01:00:00]: {long_text}\n\n"
        "**Bob** [12:10]: I agree with the revised plan.\n"
    )

    chunks = chunk_circleback_markdown(markdown)

    alice_chunks = [chunk for chunk in chunks if "Sentence " in chunk]

    assert len(alice_chunks) > 1

    for chunk in alice_chunks:
        assert chunk.startswith("**Alice** [01:00:00]:")

    assert any(chunk.startswith("**Bob** [12:10]:") for chunk in chunks)


def test_long_utterance_respects_chunk_token_budget():
    from app.config import config

    chunker = config.chunker_instance
    speech = " ".join(f"word{i}" for i in range(900))
    markdown = (
        f"# Release Planning\n\n## Transcript\n\n**Alice** [01:00:00]: {speech}\n"
    )

    chunks = chunk_circleback_markdown(markdown)
    alice_chunks = [
        chunk for chunk in chunks if chunk.startswith("**Alice** [01:00:00]:")
    ]

    assert len(alice_chunks) > 1

    oversized = [
        (index, chunker.tokenizer.count_tokens(chunk))
        for index, chunk in enumerate(alice_chunks)
        if chunker.tokenizer.count_tokens(chunk) > chunker.chunk_size
    ]

    assert not oversized, f"Chunks exceed {chunker.chunk_size} tokens: {oversized}"

    recovered = " ".join(
        chunk.removeprefix("**Alice** [01:00:00]: ").strip() for chunk in alice_chunks
    )

    assert recovered.split() == speech.split()


def test_long_chinese_utterance_preserves_every_character():
    from app.config import config

    speech = "下一阶段需要完成引用溯源和权限校验。" * 130
    prefix = "**张三** [05:20]: "
    markdown = f"# 项目周会\n\n## Transcript\n\n{prefix}{speech}\n"

    chunks = chunk_circleback_markdown(markdown)
    transcript_chunks = [chunk for chunk in chunks if chunk.startswith(prefix)]

    assert len(transcript_chunks) > 1
    assert all(
        config.chunker_instance.tokenizer.count_tokens(chunk)
        <= config.chunker_instance.chunk_size
        for chunk in transcript_chunks
    )

    recovered = "".join(chunk.removeprefix(prefix) for chunk in transcript_chunks)

    assert recovered == speech


def test_speaker_evidence_does_not_leak_between_people():
    alice_text = "ALICE_PRIVATE_POINT " * 150
    bob_text = "BOB_PRIVATE_POINT " * 150

    markdown = (
        "# Planning\n\n"
        "## Transcript\n\n"
        f"**Alice** [01:20]: {alice_text}\n\n"
        f"**Bob** [02:30]: {bob_text}\n"
    )

    chunks = chunk_circleback_markdown(markdown)

    alice_chunks = [chunk for chunk in chunks if "ALICE_PRIVATE_POINT" in chunk]
    bob_chunks = [chunk for chunk in chunks if "BOB_PRIVATE_POINT" in chunk]

    assert alice_chunks
    assert bob_chunks

    assert all(
        chunk.startswith("**Alice** [01:20]:") and "BOB_PRIVATE_POINT" not in chunk
        for chunk in alice_chunks
    )

    assert all(
        chunk.startswith("**Bob** [02:30]:") and "ALICE_PRIVATE_POINT" not in chunk
        for chunk in bob_chunks
    )


def test_meeting_notes_and_action_items_are_preserved():
    markdown = (
        "# 项目周会\n\n"
        "## Meeting Notes\n\n"
        "独立会议纪要：下周完成演示。\n\n"
        "## Action Items\n\n"
        "待办事项：准备客户验收材料。\n\n"
        "## Transcript\n\n"
        "**张三** [03:10]: 我负责知识库建设。\n"
    )

    chunks = chunk_circleback_markdown(markdown)

    assert any("独立会议纪要" in chunk for chunk in chunks)
    assert any("待办事项" in chunk for chunk in chunks)
    assert any(
        chunk.startswith("**张三** [03:10]:") and "知识库建设" in chunk
        for chunk in chunks
    )


@pytest.mark.asyncio
async def test_circleback_embeddings_use_attributed_chunk_text(
    monkeypatch,
):
    from app.tasks.document_processors import circleback_chunking

    captured = {}

    def fake_embed_texts(texts):
        captured["texts"] = list(texts)
        return [[float(i + 1)] * 384 for i in range(len(texts))]

    monkeypatch.setattr(
        circleback_chunking,
        "embed_texts",
        fake_embed_texts,
    )

    markdown = (
        "# Sprint Planning\n\n"
        "## Transcript\n\n"
        "**Alice** [01:23]: We must fix the citation issue.\n\n"
        "**Bob** [02:15]: I will verify the database.\n"
    )

    expected_texts = circleback_chunking.chunk_circleback_markdown(markdown)

    chunks = await circleback_chunking.create_circleback_document_chunks(markdown)

    assert captured["texts"] == expected_texts

    assert len(chunks) == len(expected_texts)

    for i, chunk in enumerate(chunks):
        assert chunk.content == expected_texts[i]
        assert chunk.position == i
        assert list(chunk.embedding) == [float(i + 1)] * 384

    assert any(chunk.content.startswith("**Alice** [01:23]:") for chunk in chunks)

    assert any(chunk.content.startswith("**Bob** [02:15]:") for chunk in chunks)
