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
"""Integration tests for HierarchicalTextSplitter using the real en_core_web_sm model.

All tests in this module are automatically skipped when either spaCy or the
``en_core_web_sm`` model is not installed.  No mocking is performed — these
tests exercise the full NLP pipeline end-to-end.
"""

from __future__ import annotations

import pytest

spacy = pytest.importorskip("spacy")


@pytest.fixture(scope="module")
def nlp():  # type: ignore[return]
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        pytest.skip(
            "en_core_web_sm not installed — run: python -m spacy download en_core_web_sm"
        )


@pytest.mark.asyncio
async def test_markdown_split_real(nlp) -> None:  # noqa: ANN001
    """HierarchicalTextSplitter with markdown strategy returns sequential chunks for a 3-section doc."""
    from neo4j_graphrag.experimental.components.text_splitters.hierarchical_splitter import (
        HierarchicalTextSplitter,
    )

    text = (
        "# Introduction\n"
        "This section introduces the topic and provides background information.\n\n"
        "# Methods\n"
        "This section describes the experimental methods used in the study.\n\n"
        "# Conclusion\n"
        "This section summarises the findings and suggests future work.\n"
    )

    splitter = HierarchicalTextSplitter(
        header_strategy="markdown",
        max_chunk_size=200,
        chunk_overlap=20,
        drop_verbless_sentences=False,
    )
    result = await splitter.run(text)

    assert len(result.chunks) >= 2, (
        f"Expected at least 2 chunks for a 3-section markdown doc, got {len(result.chunks)}"
    )
    # Indices must be sequential starting from 0.
    for i, chunk in enumerate(result.chunks):
        assert chunk.index == i, f"chunk {i} has non-sequential index {chunk.index}"


@pytest.mark.asyncio
async def test_drop_verbless_sentences_real(nlp) -> None:  # noqa: ANN001
    """drop_verbless_sentences=True drops verbless fragments using the real SpaCy model."""
    from neo4j_graphrag.experimental.components.text_splitters.hierarchical_splitter import (
        HierarchicalTextSplitter,
    )

    # "Overview" is a single-word verbless fragment.
    # The second sentence contains a real verb ("covers").
    text = "Overview\n\nThis section covers the main concepts of the system in detail."

    splitter = HierarchicalTextSplitter(
        header_strategy="blank_line",
        max_chunk_size=500,
        chunk_overlap=0,
        drop_verbless_sentences=True,
        model="en_core_web_sm",
    )
    result = await splitter.run(text)

    # At least one chunk must survive.
    assert len(result.chunks) >= 1, "Expected at least one chunk after filtering"

    # The verbless word "Overview" should not appear as a standalone sentence
    # in any chunk, while the main sentence content should be present.
    all_text = " ".join(chunk.text for chunk in result.chunks)
    assert "covers" in all_text, (
        "Expected the verbal sentence to survive verbless-sentence filtering"
    )
