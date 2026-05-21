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
from itertools import zip_longest
from unittest.mock import patch

import pytest
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import (
    FixedSizeSplitter,
    _adjust_chunk_end,
    _adjust_chunk_end_by_sentence,
    _adjust_chunk_start,
    _adjust_chunk_start_by_sentence,
)
from neo4j_graphrag.experimental.components.types import TextChunk


@pytest.mark.asyncio
async def test_split_text_no_overlap() -> None:
    text = "may thy knife chip and shatter"
    chunk_size = 5
    chunk_overlap = 0
    approximate = False
    splitter = FixedSizeSplitter(chunk_size, chunk_overlap, approximate)
    chunks = await splitter.run(text)
    expected_chunks = [
        TextChunk(text="may t", index=0),
        TextChunk(text="hy kn", index=1),
        TextChunk(text="ife c", index=2),
        TextChunk(text="hip a", index=3),
        TextChunk(text="nd sh", index=4),
        TextChunk(text="atter", index=5),
    ]
    for actual, expected in zip_longest(chunks.chunks, expected_chunks):
        assert actual.text == expected.text
        assert actual.index == expected.index
        assert expected.uid is not None


@pytest.mark.asyncio
async def test_split_text_with_overlap() -> None:
    text = "may thy knife chip and shatter"
    chunk_size = 10
    chunk_overlap = 2
    approximate = False
    splitter = FixedSizeSplitter(chunk_size, chunk_overlap, approximate)
    chunks = await splitter.run(text)
    expected_chunks = [
        TextChunk(text="may thy kn", index=0),
        TextChunk(text="knife chip", index=1),
        TextChunk(text="ip and sha", index=2),
        TextChunk(text="hatter", index=3),
    ]
    for actual, expected in zip_longest(chunks.chunks, expected_chunks):
        assert actual.text == expected.text
        assert actual.index == expected.index
        assert expected.uid is not None


@pytest.mark.asyncio
async def test_split_text_empty_string() -> None:
    text = ""
    chunk_size = 5
    chunk_overlap = 1
    approximate = False
    splitter = FixedSizeSplitter(chunk_size, chunk_overlap, approximate)
    chunks = await splitter.run(text)
    assert chunks.chunks == []


def test_invalid_chunk_overlap() -> None:
    with pytest.raises(ValueError) as excinfo:
        FixedSizeSplitter(5, 5)
    assert "chunk_overlap must be strictly less than chunk_size" in str(excinfo)


def test_invalid_chunk_size() -> None:
    with pytest.raises(ValueError) as excinfo:
        FixedSizeSplitter(0, 0)
    assert "chunk_size must be strictly greater than 0" in str(excinfo)


@pytest.mark.parametrize(
    "text, approximate_start, expected_start",
    [
        # Case: approximate_start is at word boundary already
        ("Hello World", 6, 6),
        # Case: approximate_start is at a whitespace already
        ("Hello World", 5, 5),
        # Case: approximate_start is at the middle of word and no whitespace is found
        ("Hello World", 2, 2),
        # Case: approximate_start is at the middle of a word
        ("Hello World", 8, 6),
        # Case: approximate_start = 0
        ("Hello World", 0, 0),
    ],
)
def test_adjust_chunk_start(
    text: str, approximate_start: int, expected_start: int
) -> None:
    """
    Test that the _adjust_chunk_start function correctly shifts
    the start index to avoid breaking words, unless no whitespace is found.
    """
    result = _adjust_chunk_start(text, approximate_start)
    assert result == expected_start


@pytest.mark.parametrize(
    "text, start, approximate_end, expected_end",
    [
        # Case: approximate_end is at word boundary already
        ("Hello World", 0, 5, 5),
        # Case: approximate_end is at the middle of a word
        ("Hello World", 0, 8, 6),
        # Case: approximate_end is at the middle of word and no whitespace is found
        ("Hello World", 0, 3, 3),
        # Case: adjusted_end == start => fallback to approximate_end
        ("Hello World", 6, 7, 7),
        # Case: end>=len(text)
        ("Hello World", 6, 15, 15),
    ],
)
def test_adjust_chunk_end(
    text: str, start: int, approximate_end: int, expected_end: int
) -> None:
    """
    Test that the _adjust_chunk_end function correctly shifts
    the end index to avoid breaking words, unless no whitespace is found.
    """
    result = _adjust_chunk_end(text, start, approximate_end)
    assert result == expected_end


@pytest.mark.parametrize(
    "sentence_starts, approximate_start, expected_start",
    [
        # approximate_start is exactly at a sentence start
        ([0, 14, 30], 14, 14),
        # approximate_start is inside a sentence — snap back to sentence start
        ([0, 14, 30], 20, 14),
        # approximate_start is before all sentence starts — no snap
        ([5, 14, 30], 2, 2),
        # empty sentence list — passthrough
        ([], 10, 10),
        # approximate_start == 0
        ([0, 14], 0, 0),
    ],
)
def test_adjust_chunk_start_by_sentence(
    sentence_starts: list[int],
    approximate_start: int,
    expected_start: int,
) -> None:
    assert _adjust_chunk_start_by_sentence(sentence_starts, approximate_start) == expected_start


@pytest.mark.parametrize(
    "sentence_ends, start, approximate_end, expected_end",
    [
        # approximate_end is exactly at a sentence end
        ([13, 29, 45], 0, 29, 29),
        # approximate_end is inside a sentence — snap back to previous sentence end
        ([13, 29, 45], 0, 35, 29),
        # only sentence end available is <= start — fallback to approximate_end
        ([5], 10, 20, 20),
        # empty sentence list — passthrough
        ([], 0, 20, 20),
        # approximate_end is past all sentence ends — return last sentence end
        ([13, 29], 0, 50, 29),
    ],
)
def test_adjust_chunk_end_by_sentence(
    sentence_ends: list[int],
    start: int,
    approximate_end: int,
    expected_end: int,
) -> None:
    assert _adjust_chunk_end_by_sentence(sentence_ends, start, approximate_end) == expected_end


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "spans, chunk_size, chunk_overlap, expected_chunks",
    [
        # Each sentence fits in its own chunk
        # text = "Hello, world. This is nltk. Last sentence." (42 chars)
        # sentence spans: [0,13), [14,27), [28,42)
        (
            [(0, 13), (14, 27), (28, 42)],
            20,
            0,
            ["Hello, world.", "This is nltk.", "Last sentence."],
        ),
        # chunk_size large enough to fit two sentences; third goes to its own chunk
        (
            [(0, 13), (14, 27), (28, 42)],
            30,
            0,
            ["Hello, world. This is nltk.", "Last sentence."],
        ),
        # with overlap: start snaps back to the sentence containing approximate_start
        (
            [(0, 13), (14, 27), (28, 42)],
            20,
            5,
            ["Hello, world.", "This is nltk.", "Last sentence."],
        ),
        # sentence longer than chunk_size — falls back to fixed-size cuts
        (
            [(0, 42)],
            20,
            0,
            ["Hello, world. This i", "s nltk. Last sentenc", "e."],
        ),
    ],
)
async def test_fixed_size_splitter_sentence_boundaries(
    spans: list[tuple[int, int]],
    chunk_size: int,
    chunk_overlap: int,
    expected_chunks: list[str],
) -> None:
    text = "Hello, world. This is nltk. Last sentence."
    with patch(
        "neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter._get_sentence_spans",
        return_value=spans,
    ):
        splitter = FixedSizeSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            sentence_boundaries=True,
        )
        result = await splitter.run(text)

    assert len(result.chunks) == len(expected_chunks)
    for i, expected_text in enumerate(expected_chunks):
        assert result.chunks[i].text == expected_text
        assert result.chunks[i].index == i


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text, chunk_size, chunk_overlap, approximate, expected_chunks",
    [
        # Case: approximate fixed size splitting
        (
            "Hello World, this is a test message.",
            10,
            2,
            True,
            ["Hello ", "World, ", "this is a ", "a test ", "message."],
        ),
        # Case: fixed size splitting
        (
            "Hello World, this is a test message.",
            10,
            2,
            False,
            ["Hello Worl", "rld, this ", "s is a tes", "est messag", "age."],
        ),
        # Case: short text => only one chunk
        (
            "Short text",
            20,
            5,
            True,
            ["Short text"],
        ),
        # Case: short text => only one chunk
        (
            "Short text",
            12,
            4,
            True,
            ["Short text"],
        ),
        # Case: text with no spaces
        (
            "1234567890",
            5,
            1,
            True,
            ["12345", "56789", "90"],
        ),
    ],
)
async def test_fixed_size_splitter_run(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    approximate: bool,
    expected_chunks: list[str],
) -> None:
    """
    Test that 'FixedSizeSplitter.run' returns the expected chunks
    for different configurations.
    """
    splitter = FixedSizeSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        approximate=approximate,
    )
    text_chunks = await splitter.run(text)

    # Verify number of chunks
    assert len(text_chunks.chunks) == len(expected_chunks)

    # Verify content of each chunk
    for i, expected_text in enumerate(expected_chunks):
        assert text_chunks.chunks[i].text == expected_text
        assert isinstance(text_chunks.chunks[i], TextChunk)
        assert text_chunks.chunks[i].index == i
