"""Verify Circleback evidence locations in source Markdown."""

import pytest

from app.indexing_pipeline.circleback_evidence import (
    circleback_chunks_with_lines,
)

pytestmark = pytest.mark.unit


def test_multiline_speech_has_correct_source_lines():
    markdown = (
        "# Weekly Meeting\n"  # 1
        "\n"  # 2
        "## Meeting Notes\n"  # 3
        "\n"  # 4
        "The team reviewed deployment risks.\n"  # 5
        "\n"  # 6
        "## Transcript\n"  # 7
        "\n"  # 8
        "**Alice** [01:00]: First deployment risk.\n"  # 9
        "Second deployment risk discussed.\n"  # 10
        "\n"  # 11
        "**Bob** [02:30]: I will verify the release.\n"  # 12
    )

    chunks = circleback_chunks_with_lines(markdown)

    alice = [c for c in chunks if c.text.startswith("**Alice** [01:00]:")]
    bob = [c for c in chunks if c.text.startswith("**Bob** [02:30]:")]

    assert alice
    assert bob

    assert all(c.start_line == 9 for c in alice)
    assert max(c.end_line for c in alice) == 10

    assert all(c.start_line == 12 and c.end_line == 12 for c in bob)


def test_repeated_speech_maps_to_correct_speakers():
    repeated = "We agree to review the deployment risk."

    markdown = (
        "# Review\n"  # 1
        "\n"  # 2
        "## Transcript\n"  # 3
        "\n"  # 4
        f"**Alice** [00:10]: {repeated}\n"  # 5
        "\n"  # 6
        f"**Bob** [00:20]: {repeated}\n"  # 7
    )

    chunks = circleback_chunks_with_lines(markdown)

    alice = [c for c in chunks if c.text.startswith("**Alice** [00:10]:")]
    bob = [c for c in chunks if c.text.startswith("**Bob** [00:20]:")]

    assert alice and bob
    assert all(c.start_line == 5 for c in alice)
    assert all(c.start_line == 7 for c in bob)
