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
"""Integration tests for SpacyEntityRelationExtractor using the real en_core_web_sm model.

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
async def test_run_returns_populated_graph(nlp) -> None:  # noqa: ANN001
    """SpacyEntityRelationExtractor.run() returns a Neo4jGraph with nodes and relationships."""
    from neo4j_graphrag.experimental.components.spacy_entity_relation_extractor import (
        SpacyEntityRelationExtractor,
    )
    from neo4j_graphrag.experimental.components.types import TextChunk, TextChunks

    extractor = SpacyEntityRelationExtractor(
        model="en_core_web_sm",
        use_linear_extractor=True,
        create_lexical_graph=False,
    )
    chunks = TextChunks(
        chunks=[
            TextChunk(
                text="Apple acquired Beats Electronics. Tim Cook announced the deal.",
                index=0,
            )
        ]
    )
    result = await extractor.run(chunks=chunks)

    assert len(result.nodes) >= 1, "Expected at least one entity node"
    assert len(result.relationships) >= 1, "Expected at least one relationship"


@pytest.mark.asyncio
async def test_confidence_values_in_range(nlp) -> None:  # noqa: ANN001
    """All relationship confidence values are floats in [0, 1]."""
    from neo4j_graphrag.experimental.components.spacy_entity_relation_extractor import (
        SpacyEntityRelationExtractor,
    )
    from neo4j_graphrag.experimental.components.types import TextChunk, TextChunks

    extractor = SpacyEntityRelationExtractor(
        model="en_core_web_sm",
        use_linear_extractor=True,
        create_lexical_graph=False,
    )
    chunks = TextChunks(
        chunks=[
            TextChunk(
                text="Apple acquired Beats Electronics. Tim Cook announced the deal.",
                index=0,
            )
        ]
    )
    result = await extractor.run(chunks=chunks)

    for rel in result.relationships:
        conf = rel.properties.get("confidence")
        assert isinstance(conf, float), f"confidence should be float, got {type(conf)}"
        assert 0.0 <= conf <= 1.0, f"confidence out of range [0,1]: {conf}"


@pytest.mark.asyncio
async def test_use_linear_extractor_false_no_related_to(nlp) -> None:  # noqa: ANN001
    """use_linear_extractor=False produces no RELATED_TO relationships."""
    from neo4j_graphrag.experimental.components.spacy_entity_relation_extractor import (
        SpacyEntityRelationExtractor,
    )
    from neo4j_graphrag.experimental.components.types import TextChunk, TextChunks

    extractor = SpacyEntityRelationExtractor(
        model="en_core_web_sm",
        use_linear_extractor=False,
        create_lexical_graph=False,
    )
    chunks = TextChunks(
        chunks=[
            TextChunk(
                text="Apple acquired Beats Electronics. Tim Cook announced the deal.",
                index=0,
            )
        ]
    )
    result = await extractor.run(chunks=chunks)

    related_to_rels = [r for r in result.relationships if r.type == "RELATED_TO"]
    assert related_to_rels == [], (
        f"Expected no RELATED_TO relationships when use_linear_extractor=False, "
        f"but got: {related_to_rels}"
    )
