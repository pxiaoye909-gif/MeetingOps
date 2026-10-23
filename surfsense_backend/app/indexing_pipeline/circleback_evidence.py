"""Source-aware Circleback chunks for the indexing pipeline."""

import re

from app.indexing_pipeline.document_chunker import LineChunk
from app.tasks.document_processors.circleback_chunking import (
    chunk_circleback_markdown,
)

_ATTRIBUTION = re.compile(r"^\*\*.+?\*\* \[\d{2}:\d{2}(?::\d{2})?\]: ")


def circleback_chunks_with_lines(markdown: str) -> list[LineChunk]:
    """Map attributed evidence back to its original Markdown lines."""
    texts = chunk_circleback_markdown(markdown)
    cursor = 0
    results = []

    for text in texts:
        match = _ATTRIBUTION.match(text)

        # Repeated speaker/time labels are metadata added by the chunker.
        # They may not exist at the start of every substring in the source.
        original = text[match.end() :] if match else text
        needle = original.strip()

        if not needle:
            continue

        start = markdown.find(needle, cursor)

        if start < 0:
            raise ValueError("Circleback chunk cannot be located in source Markdown")

        end = start + len(needle)

        start_line = markdown.count("\n", 0, start) + 1
        end_line = markdown.count("\n", 0, end - 1) + 1

        results.append(
            LineChunk(
                text=text,
                start_line=start_line,
                end_line=end_line,
            )
        )
        cursor = end

    return results
