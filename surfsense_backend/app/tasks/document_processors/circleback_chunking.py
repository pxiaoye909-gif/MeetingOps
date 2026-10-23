"""Evidence-preserving chunking for Circleback meeting transcripts."""

import asyncio
import re

from app.config import config
from app.db import Chunk
from app.utils.document_converters import embed_texts

_TRANSCRIPT_HEADING = re.compile(r"(?m)^## Transcript[ \t]*$")
_SPEAKER_LINE = re.compile(
    r"(?m)^\*\*(?P<speaker>.+?)\*\* "
    r"\[(?P<timestamp>\d{2}:\d{2}(?::\d{2})?)\]:[ \t]*"
)


def _split_attributed_speech(
    prefix: str,
    speech: str,
    chunker,
) -> list[str]:
    """Split speech without exceeding the final attributed token budget."""
    count_tokens = chunker.tokenizer.count_tokens
    limit = chunker.chunk_size

    if count_tokens(prefix) >= limit:
        raise ValueError("Speaker and timestamp exceed the chunk token budget")

    results: list[str] = []
    current = ""

    for original_unit in re.findall(r"\S+\s*|\s+", speech):
        unit = original_unit

        while unit:
            candidate = prefix + current + unit

            if count_tokens(candidate) <= limit:
                current += unit
                break

            if current:
                results.append(prefix + current)
                current = ""
                continue

            left, right = 1, len(unit)
            best = 0

            while left <= right:
                middle = (left + right) // 2

                if count_tokens(prefix + unit[:middle]) <= limit:
                    best = middle
                    left = middle + 1
                else:
                    right = middle - 1

            if best == 0:
                raise ValueError("Cannot fit speech text within token budget")

            results.append(prefix + unit[:best])
            unit = unit[best:]

    if current:
        results.append(prefix + current)

    return results


def chunk_circleback_markdown(markdown: str) -> list[str]:
    """Keep the originating speaker and timestamp on transcript chunks."""
    chunker = config.chunker_instance

    heading = _TRANSCRIPT_HEADING.search(markdown)
    if heading is None:
        return [part.text for part in chunker.chunk(markdown)]

    transcript = markdown[heading.end() :]
    matches = list(_SPEAKER_LINE.finditer(transcript))

    if not matches:
        return [part.text for part in chunker.chunk(markdown)]

    preamble = markdown[: heading.end()].strip()
    chunks = [part.text for part in chunker.chunk(preamble)] if preamble else []

    for index, match in enumerate(matches):
        end = (
            matches[index + 1].start() if index + 1 < len(matches) else len(transcript)
        )
        utterance = transcript[match.end() : end].strip()
        prefix = f"**{match.group('speaker')}** [{match.group('timestamp')}]: "

        if not utterance:
            continue

        chunks.extend(_split_attributed_speech(prefix, utterance, chunker))

    return chunks


async def create_circleback_document_chunks(
    markdown: str,
) -> list[Chunk]:
    """Create database chunks with embeddings of the attributed text."""
    texts = chunk_circleback_markdown(markdown)
    embeddings = await asyncio.to_thread(embed_texts, texts)

    if len(texts) != len(embeddings):
        raise ValueError("Circleback chunk and embedding counts differ")

    return [
        Chunk(content=text, embedding=embedding, position=index)
        for index, (text, embedding) in enumerate(zip(texts, embeddings, strict=True))
    ]
