#  Copyright (c) "Neo4j"
#  Neo4j Sweden AB [https://neo4j.com]
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#      https://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Unit tests for HierarchicalTextSplitter.

SpaCy is not downloaded in these tests.  Where the splitter would normally
load a model (`drop_verbless_sentences=True` or `header_strategy="spacy_verbless"`),
the tests patch `spacy.load` with a lightweight fake nlp object so that no
network access or model installation is required.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# Skip the entire module when spaCy is not installed at all.
spacy = pytest.importorskip("spacy")

from neo4j_graphrag.experimental.components.text_splitters.hierarchical_splitter import (  # noqa: E402
    HierarchicalTextSplitter,
)


# ---------------------------------------------------------------------------
# Helpers for building fake SpaCy objects without a real model
# ---------------------------------------------------------------------------


def _make_fake_token(text: str, pos: str) -> MagicMock:
    """Return a MagicMock that looks like a spaCy Token."""
    tok = MagicMock()
    tok.text = text
    tok.pos_ = pos
    return tok


def _make_fake_sent(text: str, tokens: list[MagicMock]) -> MagicMock:
    """Return a MagicMock that looks like a spaCy Span (sentence).

    Uses ``side_effect`` instead of ``return_value`` so that each call to
    ``iter(sent)`` creates a *fresh* iterator — important if the sentence is
    iterated more than once (e.g. across multiple ``run()`` calls or in future
    multi-pass tests).
    """
    sent = MagicMock()
    sent.text = text
    sent.__iter__ = MagicMock(side_effect=lambda: iter(tokens))
    return sent


def _make_fake_doc(sentences: list[MagicMock]) -> MagicMock:
    """Return a MagicMock that looks like a spaCy Doc with .sents."""
    doc = MagicMock()
    doc.sents = sentences
    doc.__iter__ = MagicMock(return_value=iter([]))
    return doc


def _make_nlp_returning_doc(sentences: list[MagicMock]) -> MagicMock:
    """Return a callable MagicMock that acts as a spaCy nlp() pipeline."""
    nlp = MagicMock()
    nlp.return_value = _make_fake_doc(sentences)
    return nlp


# ---------------------------------------------------------------------------
# Tests: header_strategy="markdown"
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_markdown_two_sections_produce_two_chunks() -> None:
    """Two Markdown sections produce exactly two chunks."""
    text = "# Introduction\nThis is the intro section.\n# Conclusion\nThis is the conclusion."
    splitter = HierarchicalTextSplitter(
        max_chunk_size=2048,
        chunk_overlap=0,
        header_strategy="markdown",
        drop_verbless_sentences=False,
    )
    result = await splitter.run(text)
    assert len(result.chunks) == 2
    assert "Introduction" in result.chunks[0].text
    assert "Conclusion" in result.chunks[1].text


@pytest.mark.asyncio
async def test_markdown_single_section_produces_one_chunk() -> None:
    """Text with no Markdown headers is treated as one section."""
    text = "No headers here. Just a single paragraph of text."
    splitter = HierarchicalTextSplitter(
        max_chunk_size=2048,
        chunk_overlap=0,
        header_strategy="markdown",
        drop_verbless_sentences=False,
    )
    result = await splitter.run(text)
    assert len(result.chunks) == 1


@pytest.mark.asyncio
async def test_markdown_three_sections_sequential_indices() -> None:
    """Three Markdown sections produce chunks with sequential indices 0, 1, 2."""
    text = "# A\nSection A body.\n# B\nSection B body.\n# C\nSection C body."
    splitter = HierarchicalTextSplitter(
        max_chunk_size=2048,
        chunk_overlap=0,
        header_strategy="markdown",
        drop_verbless_sentences=False,
    )
    result = await splitter.run(text)
    assert len(result.chunks) == 3
    for i, chunk in enumerate(result.chunks):
        assert chunk.index == i


# ---------------------------------------------------------------------------
# Tests: header_strategy="capitalization"
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_capitalization_allcaps_header_splits() -> None:
    """ALL_CAPS lines without terminal punctuation act as section headers."""
    text = "INTRODUCTION\nThis paragraph describes the introduction.\nCONCLUSION\nThis paragraph wraps things up."
    splitter = HierarchicalTextSplitter(
        max_chunk_size=2048,
        chunk_overlap=0,
        header_strategy="capitalization",
        drop_verbless_sentences=False,
    )
    result = await splitter.run(text)
    assert len(result.chunks) == 2
    assert "INTRODUCTION" in result.chunks[0].text
    assert "CONCLUSION" in result.chunks[1].text


@pytest.mark.asyncio
async def test_capitalization_title_case_header_splits() -> None:
    """Title Case lines without terminal punctuation act as section headers."""
    text = "First Section Title\nContent for the first section.\nSecond Section Title\nContent for the second section."
    splitter = HierarchicalTextSplitter(
        max_chunk_size=2048,
        chunk_overlap=0,
        header_strategy="capitalization",
        drop_verbless_sentences=False,
    )
    result = await splitter.run(text)
    assert len(result.chunks) == 2


@pytest.mark.asyncio
async def test_capitalization_indices_sequential() -> None:
    """Chunk indices are sequential starting from 0."""
    text = "PART ONE\nBody of part one.\nPART TWO\nBody of part two.\nPART THREE\nBody of part three."
    splitter = HierarchicalTextSplitter(
        max_chunk_size=2048,
        chunk_overlap=0,
        header_strategy="capitalization",
        drop_verbless_sentences=False,
    )
    result = await splitter.run(text)
    for i, chunk in enumerate(result.chunks):
        assert chunk.index == i, f"chunk {i} has index {chunk.index}"


# ---------------------------------------------------------------------------
# Tests: header_strategy="blank_line"
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_blank_line_short_surrounded_header_splits() -> None:
    """Short lines surrounded by blank lines act as section headers."""
    text = "\nOverview\n\nThis section covers the overview of the system.\n\nDetails\n\nThis section covers the fine-grained details."
    splitter = HierarchicalTextSplitter(
        max_chunk_size=2048,
        chunk_overlap=0,
        header_strategy="blank_line",
        drop_verbless_sentences=False,
    )
    result = await splitter.run(text)
    assert len(result.chunks) == 2
    assert "Overview" in result.chunks[0].text
    assert "Details" in result.chunks[1].text


@pytest.mark.asyncio
async def test_blank_line_indices_sequential() -> None:
    """Chunk indices are sequential starting from 0 for blank_line strategy."""
    text = "\nPart A\n\nContent for part A goes here.\n\nPart B\n\nContent for part B goes here."
    splitter = HierarchicalTextSplitter(
        max_chunk_size=2048,
        chunk_overlap=0,
        header_strategy="blank_line",
        drop_verbless_sentences=False,
    )
    result = await splitter.run(text)
    for i, chunk in enumerate(result.chunks):
        assert chunk.index == i


# ---------------------------------------------------------------------------
# Tests: overlap splitting for large sections
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_large_section_split_with_overlap() -> None:
    """A section larger than max_chunk_size is split into multiple chunks
    with the last N characters of chunk K equal to the first N characters
    of chunk K+1.

    Strategy: use no header strategy (treat whole text as one section) via a
    single Markdown section; make the body very long and max_chunk_size small
    relative to the body so that we get several chunks all in the body content,
    and the overlap check applies to non-header chunks.
    """
    # The section body is pure 'x' characters with no whitespace.
    # With max_size=50, overlap=20, step=30:
    #   chunk 0: body[0:50]   = 'x'*50
    #   chunk 1: body[30:80]  = 'x'*50, so overlap = body[30:50] = 'x'*20
    # We avoid the Markdown header prefix eating into the overlap window by
    # using a plain-text body (no Markdown header at all, so the whole text
    # is one section) and using drop_verbless_sentences=False.
    body = "x" * 500
    overlap = 20
    max_size = 50
    splitter = HierarchicalTextSplitter(
        max_chunk_size=max_size,
        chunk_overlap=overlap,
        header_strategy="markdown",
        drop_verbless_sentences=False,
    )
    result = await splitter.run(body)
    chunks = result.chunks
    assert len(chunks) > 1

    # Verify overlap: last `overlap` chars of chunk[k] == first `overlap` chars of chunk[k+1].
    for k in range(len(chunks) - 1):
        tail = chunks[k].text[-overlap:]
        head = chunks[k + 1].text[:overlap]
        assert tail == head, (
            f"Overlap mismatch between chunk {k} and {k + 1}: "
            f"tail={tail!r}, head={head!r}"
        )


@pytest.mark.asyncio
async def test_small_section_emitted_as_single_chunk() -> None:
    """A section shorter than max_chunk_size is emitted as a single chunk."""
    text = "# Tiny\nShort body."
    splitter = HierarchicalTextSplitter(
        max_chunk_size=2048,
        chunk_overlap=0,
        header_strategy="markdown",
        drop_verbless_sentences=False,
    )
    result = await splitter.run(text)
    assert len(result.chunks) == 1
    assert "Tiny" in result.chunks[0].text
    assert "Short body" in result.chunks[0].text


@pytest.mark.asyncio
async def test_overlap_chunk_indices_sequential() -> None:
    """Indices remain sequential when a section is split due to size."""
    body = "y" * 300
    text = f"# Big Section\n{body}"
    splitter = HierarchicalTextSplitter(
        max_chunk_size=60,
        chunk_overlap=10,
        header_strategy="markdown",
        drop_verbless_sentences=False,
    )
    result = await splitter.run(text)
    assert len(result.chunks) > 1
    for i, chunk in enumerate(result.chunks):
        assert chunk.index == i


# ---------------------------------------------------------------------------
# Tests: drop_verbless_sentences=True (SpaCy mocked)
# ---------------------------------------------------------------------------


def _make_spacy_nlp_with_verbless() -> MagicMock:
    """Return a fake nlp() that drops verbless sentences when called.

    Sentence 1 (verbless): "No verb here" — tokens have no VERB/AUX.
    Sentence 2 (with verb): "The dog runs fast" — one VERB token.
    """
    # Tokens for sentence 1 (verbless: no VERB/AUX tags)
    sent1_tokens = [
        _make_fake_token("No", "DET"),
        _make_fake_token("verb", "NOUN"),
        _make_fake_token("here", "ADV"),
    ]
    sent1 = _make_fake_sent("No verb here", sent1_tokens)

    # Tokens for sentence 2 (contains a VERB)
    sent2_tokens = [
        _make_fake_token("The", "DET"),
        _make_fake_token("dog", "NOUN"),
        _make_fake_token("runs", "VERB"),
        _make_fake_token("fast", "ADV"),
    ]
    sent2 = _make_fake_sent("The dog runs fast", sent2_tokens)

    return _make_nlp_returning_doc([sent1, sent2])


@pytest.mark.asyncio
async def test_drop_verbless_removes_verbless_sentence() -> None:
    """When drop_verbless_sentences=True, sentences with no verb are removed."""
    fake_nlp = _make_spacy_nlp_with_verbless()

    with patch("spacy.load", return_value=fake_nlp):
        splitter = HierarchicalTextSplitter(
            max_chunk_size=2048,
            chunk_overlap=0,
            header_strategy="markdown",
            drop_verbless_sentences=True,
            model="en_core_web_sm",
        )

    text = "# Section\nNo verb here. The dog runs fast."
    result = await splitter.run(text)

    assert len(result.chunks) == 1
    chunk_text = result.chunks[0].text
    # Verbless sentence should be gone; verbal sentence should remain.
    assert "The dog runs fast" in chunk_text
    assert "No verb here" not in chunk_text


@pytest.mark.asyncio
async def test_drop_verbless_keeps_verbal_sentences() -> None:
    """When all sentences contain a verb, no text is removed."""
    sent_tokens = [
        _make_fake_token("She", "PRON"),
        _make_fake_token("walks", "VERB"),
        _make_fake_token("home", "NOUN"),
    ]
    sent = _make_fake_sent("She walks home", sent_tokens)
    fake_nlp = _make_nlp_returning_doc([sent])

    with patch("spacy.load", return_value=fake_nlp):
        splitter = HierarchicalTextSplitter(
            max_chunk_size=2048,
            chunk_overlap=0,
            header_strategy="markdown",
            drop_verbless_sentences=True,
            model="en_core_web_sm",
        )

    text = "# Section\nShe walks home."
    result = await splitter.run(text)

    assert len(result.chunks) == 1
    assert "She walks home" in result.chunks[0].text


# ---------------------------------------------------------------------------
# Tests: chunk index sequencing across multiple sections
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_indices_sequential_across_multiple_sections() -> None:
    """Indices are global — they continue from the last chunk of the previous section."""
    text = (
        "# Alpha\nFirst section body.\n"
        "# Beta\nSecond section body.\n"
        "# Gamma\nThird section body."
    )
    splitter = HierarchicalTextSplitter(
        max_chunk_size=2048,
        chunk_overlap=0,
        header_strategy="markdown",
        drop_verbless_sentences=False,
    )
    result = await splitter.run(text)
    assert len(result.chunks) == 3
    for expected_index, chunk in enumerate(result.chunks):
        assert chunk.index == expected_index


# ---------------------------------------------------------------------------
# Tests: header_strategy="spacy_verbless"
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_spacy_verbless_strategy_splits_at_verbless_heading() -> None:
    """header_strategy="spacy_verbless" uses SpaCy to detect verbless headings.

    A short sentence with no verb that precedes a longer sentence is treated as
    a section heading and causes a split.  This test mocks the nlp pipeline so
    no model is downloaded.
    """
    # Sentence 1 (verbless heading, short ≤ 80 chars):  "Introduction"
    sent1_tokens = [_make_fake_token("Introduction", "NOUN")]
    sent1 = _make_fake_sent("Introduction", sent1_tokens)

    # Sentence 2 (long body sentence, > 80 chars, contains VERB):
    long_body = "This section covers all the foundational concepts you need to understand before proceeding further."
    sent2_tokens = [
        _make_fake_token("This", "DET"),
        _make_fake_token("section", "NOUN"),
        _make_fake_token("covers", "VERB"),
    ]
    sent2 = _make_fake_sent(long_body, sent2_tokens)

    # Sentence 3 (verbless heading): "Conclusion"
    sent3_tokens = [_make_fake_token("Conclusion", "NOUN")]
    sent3 = _make_fake_sent("Conclusion", sent3_tokens)

    # Sentence 4 (another long body):
    long_body2 = "This final section wraps up all the topics discussed and provides closing remarks for the reader."
    sent4_tokens = [
        _make_fake_token("This", "DET"),
        _make_fake_token("section", "NOUN"),
        _make_fake_token("wraps", "VERB"),
    ]
    sent4 = _make_fake_sent(long_body2, sent4_tokens)

    fake_nlp = _make_nlp_returning_doc([sent1, sent2, sent3, sent4])

    with patch("spacy.load", return_value=fake_nlp):
        splitter = HierarchicalTextSplitter(
            max_chunk_size=2048,
            chunk_overlap=0,
            header_strategy="spacy_verbless",
            drop_verbless_sentences=False,
            model="en_core_web_sm",
        )

    # Use arbitrary text — the nlp mock controls what sents are returned.
    text = "Introduction. " + long_body + " Conclusion. " + long_body2
    result = await splitter.run(text)

    # Two verbless headings (sent1, sent3) each trigger a split → 2 sections.
    assert len(result.chunks) == 2
    assert result.chunks[0].index == 0
    assert result.chunks[1].index == 1


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_text_returns_no_chunks() -> None:
    """Empty input produces an empty chunk list."""
    splitter = HierarchicalTextSplitter(
        max_chunk_size=2048,
        chunk_overlap=0,
        header_strategy="markdown",
        drop_verbless_sentences=False,
    )
    result = await splitter.run("")
    assert result.chunks == []


@pytest.mark.asyncio
async def test_whitespace_only_returns_no_chunks() -> None:
    """Whitespace-only input produces an empty chunk list."""
    splitter = HierarchicalTextSplitter(
        max_chunk_size=2048,
        chunk_overlap=0,
        header_strategy="markdown",
        drop_verbless_sentences=False,
    )
    result = await splitter.run("   \n\t\n  ")
    assert result.chunks == []


def test_invalid_header_strategy_raises() -> None:
    """An unrecognised header_strategy raises ValueError at construction time."""
    with pytest.raises(ValueError, match="header_strategy must be one of"):
        HierarchicalTextSplitter(
            max_chunk_size=2048,
            chunk_overlap=0,
            header_strategy="unknown_strategy",
            drop_verbless_sentences=False,
        )


def test_chunk_overlap_ge_max_chunk_size_raises() -> None:
    """chunk_overlap >= max_chunk_size raises ValueError."""
    with pytest.raises(ValueError, match="chunk_overlap must be strictly less than max_chunk_size"):
        HierarchicalTextSplitter(
            max_chunk_size=100,
            chunk_overlap=100,
            header_strategy="markdown",
            drop_verbless_sentences=False,
        )


def test_max_chunk_size_zero_raises() -> None:
    """max_chunk_size=0 raises ValueError."""
    with pytest.raises(ValueError, match="max_chunk_size must be strictly greater than 0"):
        HierarchicalTextSplitter(
            max_chunk_size=0,
            chunk_overlap=0,
            header_strategy="markdown",
            drop_verbless_sentences=False,
        )
