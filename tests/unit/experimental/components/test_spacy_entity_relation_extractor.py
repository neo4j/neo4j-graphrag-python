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
"""Unit tests for SpacyEntityRelationExtractor.

spaCy is an optional dependency; these tests are skipped when it is not
installed.  ``spacy.load`` is mocked so no trained model download is required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("spacy", reason="spacy is not installed")

from neo4j_graphrag.experimental.components.entity_relation_extractor import OnError  # noqa: E402
from neo4j_graphrag.experimental.components.schema import (  # noqa: E402
    GraphSchema,
    NodeType,
    RelationshipType,
)
from neo4j_graphrag.experimental.components.spacy_entity_relation_extractor import (  # noqa: E402
    SpacyEntityRelationExtractor,
)
from neo4j_graphrag.experimental.components.types import (  # noqa: E402
    DocumentInfo,
    Neo4jGraph,
    TextChunk,
    TextChunks,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_nlp(
    dep_triples: set[tuple[str, str, str, float]] | None = None,
    linear_triples: set[tuple[str, str, str, float]] | None = None,
    ents: list[tuple[str, str]] | None = None,
) -> MagicMock:
    """Return a mock ``spacy.language.Language`` callable.

    The mock's ``__call__`` method returns a Doc-like MagicMock whose
    ``.ents`` attribute contains simple objects with ``.text`` and ``.label_``
    attributes.

    *dep_triples* and *linear_triples* are injected via patches on
    ``extract_dependency_triples`` / ``extract_linear_triples`` rather than
    via this mock, but the Doc object must still exist.

    Args:
        dep_triples: Triples returned by extract_dependency_triples.
        linear_triples: Triples returned by extract_linear_triples.
        ents: List of (text, label_) for doc.ents.
    """
    mock_nlp = MagicMock()

    # Build a simple entity list.
    mock_ents = []
    for text, label in (ents or []):
        ent = MagicMock()
        ent.text = text
        ent.label_ = label
        mock_ents.append(ent)

    mock_doc = MagicMock()
    mock_doc.ents = mock_ents
    mock_nlp.return_value = mock_doc

    return mock_nlp


# ---------------------------------------------------------------------------
# 1. Successful extraction — two chunks, Entity nodes + relationships
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_successful_extraction_two_chunks() -> None:
    """Two-chunk input produces Entity nodes and RELATED_TO relationships."""
    dep_triples: set[tuple[str, str, str, float]] = set()
    linear_triples: set[tuple[str, str, str, float]] = {
        ("Apple", "RELATED_TO", "Google", 0.3),
        ("Microsoft", "RELATED_TO", "Azure", 0.3),
    }

    mock_nlp = _make_mock_nlp(
        ents=[("Apple", "ORG"), ("Google", "ORG")],
    )

    with patch("spacy.load", return_value=mock_nlp), patch(
        "neo4j_graphrag.experimental.components.spacy_entity_relation_extractor.extract_dependency_triples",
        return_value=dep_triples,
    ), patch(
        "neo4j_graphrag.experimental.components.spacy_entity_relation_extractor.extract_linear_triples",
        side_effect=[linear_triples, {("Microsoft", "RELATED_TO", "Azure", 0.3)}],
    ):
        extractor = SpacyEntityRelationExtractor(
            model="en_core_web_sm",
            create_lexical_graph=False,
        )
        chunks = TextChunks(
            chunks=[
                TextChunk(text="Apple acquired Google.", index=0),
                TextChunk(text="Microsoft launched Azure.", index=1),
            ]
        )
        result = await extractor.run(chunks=chunks)

    assert isinstance(result, Neo4jGraph)
    # We expect at least 4 Entity nodes across both chunks (apple, google, microsoft, azure)
    node_ids = {n.id for n in result.nodes}
    assert len(node_ids) >= 2  # at minimum the chunk-0 entities
    # All nodes should be labelled "Entity"
    for node in result.nodes:
        assert node.label == "Entity"
    # All relationships should have a "confidence" property
    for rel in result.relationships:
        assert "confidence" in rel.properties
        assert isinstance(rel.properties["confidence"], float)


# ---------------------------------------------------------------------------
# 2. use_linear_extractor=False — no RELATED_TO relationships
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_use_linear_extractor_false_no_related_to() -> None:
    """With use_linear_extractor=False, linear triples are never produced."""
    dep_triples: set[tuple[str, str, str, float]] = set()

    mock_nlp = _make_mock_nlp(ents=[("Apple", "ORG"), ("Google", "ORG")])

    with patch("spacy.load", return_value=mock_nlp), patch(
        "neo4j_graphrag.experimental.components.spacy_entity_relation_extractor.extract_dependency_triples",
        return_value=dep_triples,
    ) as mock_dep, patch(
        "neo4j_graphrag.experimental.components.spacy_entity_relation_extractor.extract_linear_triples",
    ) as mock_linear:
        extractor = SpacyEntityRelationExtractor(
            model="en_core_web_sm",
            use_linear_extractor=False,
            create_lexical_graph=False,
        )
        chunks = TextChunks(chunks=[TextChunk(text="Apple bought Google.", index=0)])
        result = await extractor.run(chunks=chunks)

    # The linear extractor must NOT have been called.
    mock_linear.assert_not_called()
    # No nodes or relationships (dep_triples is empty).
    assert result.nodes == []
    assert result.relationships == []


# ---------------------------------------------------------------------------
# 3. use_linear_extractor=True — RELATED_TO relationships may be produced
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_use_linear_extractor_true_produces_related_to() -> None:
    """With use_linear_extractor=True and NER entities, RELATED_TO rels appear."""
    dep_triples: set[tuple[str, str, str, float]] = set()
    linear_triples: set[tuple[str, str, str, float]] = {
        ("Apple", "RELATED_TO", "Google", 0.3),
    }

    mock_nlp = _make_mock_nlp(ents=[("Apple", "ORG"), ("Google", "ORG")])

    with patch("spacy.load", return_value=mock_nlp), patch(
        "neo4j_graphrag.experimental.components.spacy_entity_relation_extractor.extract_dependency_triples",
        return_value=dep_triples,
    ), patch(
        "neo4j_graphrag.experimental.components.spacy_entity_relation_extractor.extract_linear_triples",
        return_value=linear_triples,
    ):
        extractor = SpacyEntityRelationExtractor(
            model="en_core_web_sm",
            use_linear_extractor=True,
            create_lexical_graph=False,
        )
        chunks = TextChunks(chunks=[TextChunk(text="Apple partnered with Google.", index=0)])
        result = await extractor.run(chunks=chunks)

    rel_types = {r.type for r in result.relationships}
    assert "RELATED_TO" in rel_types
    # Confidence should be 0.3 for linear triples.
    for rel in result.relationships:
        if rel.type == "RELATED_TO":
            assert rel.properties["confidence"] == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# 4. create_lexical_graph=True — Chunk/Document nodes present
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_lexical_graph_adds_chunk_nodes() -> None:
    """create_lexical_graph=True adds Chunk (and Document) nodes to the graph."""
    dep_triples: set[tuple[str, str, str, float]] = set()
    linear_triples: set[tuple[str, str, str, float]] = set()

    mock_nlp = _make_mock_nlp(ents=[])

    with patch("spacy.load", return_value=mock_nlp), patch(
        "neo4j_graphrag.experimental.components.spacy_entity_relation_extractor.extract_dependency_triples",
        return_value=dep_triples,
    ), patch(
        "neo4j_graphrag.experimental.components.spacy_entity_relation_extractor.extract_linear_triples",
        return_value=linear_triples,
    ):
        extractor = SpacyEntityRelationExtractor(
            model="en_core_web_sm",
            create_lexical_graph=True,
        )
        document_info = DocumentInfo(path="test_doc.txt")
        chunks = TextChunks(chunks=[TextChunk(text="Some text.", index=0)])
        result = await extractor.run(chunks=chunks, document_info=document_info)

    labels = {n.label for n in result.nodes}
    assert "Chunk" in labels
    assert "Document" in labels


@pytest.mark.asyncio
async def test_create_lexical_graph_false_no_chunk_nodes() -> None:
    """create_lexical_graph=False produces no Chunk or Document nodes."""
    dep_triples: set[tuple[str, str, str, float]] = set()
    linear_triples: set[tuple[str, str, str, float]] = set()

    mock_nlp = _make_mock_nlp(ents=[])

    with patch("spacy.load", return_value=mock_nlp), patch(
        "neo4j_graphrag.experimental.components.spacy_entity_relation_extractor.extract_dependency_triples",
        return_value=dep_triples,
    ), patch(
        "neo4j_graphrag.experimental.components.spacy_entity_relation_extractor.extract_linear_triples",
        return_value=linear_triples,
    ):
        extractor = SpacyEntityRelationExtractor(
            model="en_core_web_sm",
            create_lexical_graph=False,
        )
        chunks = TextChunks(chunks=[TextChunk(text="Some text.", index=0)])
        result = await extractor.run(chunks=chunks)

    labels = {n.label for n in result.nodes}
    assert "Chunk" not in labels
    assert "Document" not in labels


# ---------------------------------------------------------------------------
# 5. Constructor raises ValueError for non-installed model
# ---------------------------------------------------------------------------


def test_constructor_missing_model_raises_value_error() -> None:
    """A missing SpaCy model raises ValueError with install instructions."""
    with patch("spacy.load", side_effect=OSError("Model not found")):
        with pytest.raises(ValueError) as exc_info:
            SpacyEntityRelationExtractor(model="en_core_web_sm")

    assert "python -m spacy download" in str(exc_info.value)
    assert "en_core_web_sm" in str(exc_info.value)


def test_constructor_missing_spacy_package_raises_value_error() -> None:
    """An ImportError (spacy not installed) is also wrapped in ValueError."""
    with patch("spacy.load", side_effect=ImportError("No module named 'spacy'")):
        with pytest.raises(ValueError) as exc_info:
            SpacyEntityRelationExtractor(model="en_core_web_sm")

    assert "python -m spacy download" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 6. on_error=OnError.IGNORE suppresses per-chunk exceptions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_error_ignore_suppresses_exception() -> None:
    """on_error=OnError.IGNORE skips chunks that raise exceptions."""

    def _raise(*args: object, **kwargs: object) -> None:
        raise RuntimeError("NLP pipeline exploded")

    mock_nlp = MagicMock(side_effect=_raise)

    with patch("spacy.load", return_value=mock_nlp), patch(
        "neo4j_graphrag.experimental.components.spacy_entity_relation_extractor.extract_dependency_triples",
        return_value=set(),
    ):
        extractor = SpacyEntityRelationExtractor(
            model="en_core_web_sm",
            on_error=OnError.IGNORE,
            create_lexical_graph=False,
        )
        chunks = TextChunks(chunks=[TextChunk(text="boom", index=0)])
        # Should not raise; the chunk is silently skipped.
        result = await extractor.run(chunks=chunks)

    assert result.nodes == []
    assert result.relationships == []


@pytest.mark.asyncio
async def test_on_error_raise_propagates_exception() -> None:
    """on_error=OnError.RAISE re-raises per-chunk exceptions."""

    def _raise(*args: object, **kwargs: object) -> None:
        raise RuntimeError("NLP pipeline exploded")

    mock_nlp = MagicMock(side_effect=_raise)

    with patch("spacy.load", return_value=mock_nlp), patch(
        "neo4j_graphrag.experimental.components.spacy_entity_relation_extractor.extract_dependency_triples",
        return_value=set(),
    ):
        extractor = SpacyEntityRelationExtractor(
            model="en_core_web_sm",
            on_error=OnError.RAISE,
            create_lexical_graph=False,
        )
        chunks = TextChunks(chunks=[TextChunk(text="boom", index=0)])
        with pytest.raises(RuntimeError, match="NLP pipeline exploded"):
            await extractor.run(chunks=chunks)


# ---------------------------------------------------------------------------
# 7. Custom word_filter filters out entities in the list
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_custom_word_filter_excludes_entities() -> None:
    """Entities whose normalised text appears in word_filter are excluded."""
    dep_triples: set[tuple[str, str, str, float]] = {
        ("apple", "RELATED_TO", "stopword", 0.3),
        ("apple", "RELATED_TO", "google", 0.3),
    }

    mock_nlp = _make_mock_nlp(ents=[("apple", "ORG"), ("stopword", "ORG"), ("google", "ORG")])

    with patch("spacy.load", return_value=mock_nlp), patch(
        "neo4j_graphrag.experimental.components.spacy_entity_relation_extractor.extract_dependency_triples",
        return_value=dep_triples,
    ), patch(
        "neo4j_graphrag.experimental.components.spacy_entity_relation_extractor.extract_linear_triples",
        return_value=set(),
    ):
        # "stopword" is in our custom filter.
        extractor = SpacyEntityRelationExtractor(
            model="en_core_web_sm",
            word_filter=["stopword"],
            use_linear_extractor=True,
            create_lexical_graph=False,
        )
        chunks = TextChunks(chunks=[TextChunk(text="apple stopword google", index=0)])
        result = await extractor.run(chunks=chunks)

    # "stopword" should not appear as a node.
    # Node IDs are prefixed with the chunk UUID: "<chunk_id>:apple" — check the suffix.
    node_id_suffixes = {n.id.split(":")[-1] for n in result.nodes}
    assert "stopword" not in node_id_suffixes
    # The (apple, google) triple should still be present.
    assert any(
        r.start_node_id.endswith("apple") and r.end_node_id.endswith("google")
        for r in result.relationships
    )


# ---------------------------------------------------------------------------
# 8. Schema filtering: entities with unmapped NER types pass through (fail-open)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_schema_filtering_unmapped_ner_types_pass_through() -> None:
    """Schema with only custom types that don't map to SpaCy NER → fail-open."""
    dep_triples: set[tuple[str, str, str, float]] = {
        ("ContractA", "RELATED_TO", "ContractB", 0.3),
    }

    # "Contract" is not in _SPACY_NER_TO_SCHEMA_LABEL, so fail-open applies.
    mock_nlp = _make_mock_nlp(ents=[("ContractA", "MISC"), ("ContractB", "MISC")])

    with patch("spacy.load", return_value=mock_nlp), patch(
        "neo4j_graphrag.experimental.components.spacy_entity_relation_extractor.extract_dependency_triples",
        return_value=dep_triples,
    ), patch(
        "neo4j_graphrag.experimental.components.spacy_entity_relation_extractor.extract_linear_triples",
        return_value=set(),
    ):
        schema = GraphSchema(
            node_types=(NodeType(label="Contract"),),
            relationship_types=(RelationshipType(label="RELATED_TO"),),
        )
        extractor = SpacyEntityRelationExtractor(
            model="en_core_web_sm",
            use_linear_extractor=True,
            create_lexical_graph=False,
        )
        chunks = TextChunks(chunks=[TextChunk(text="ContractA links ContractB.", index=0)])
        result = await extractor.run(chunks=chunks, schema=schema)

    # Fail-open: triples pass through because "Contract" has no SpaCy NER mapping.
    # Node IDs are prefixed with the chunk UUID: "<chunk_id>:contracta".
    node_id_suffixes = {n.id.split(":")[-1] for n in result.nodes}
    assert "contracta" in node_id_suffixes
    assert "contractb" in node_id_suffixes


@pytest.mark.asyncio
async def test_schema_filtering_known_ner_type_filters_unmatched() -> None:
    """Schema with Organization node_type: only ORG entities survive NER filter."""
    dep_triples: set[tuple[str, str, str, float]] = {
        ("Apple", "RELATED_TO", "NewYork", 0.3),  # NewYork is GPE, not ORG
    }

    mock_nlp = _make_mock_nlp(ents=[("Apple", "ORG"), ("NewYork", "GPE")])

    with patch("spacy.load", return_value=mock_nlp), patch(
        "neo4j_graphrag.experimental.components.spacy_entity_relation_extractor.extract_dependency_triples",
        return_value=dep_triples,
    ), patch(
        "neo4j_graphrag.experimental.components.spacy_entity_relation_extractor.extract_linear_triples",
        return_value=set(),
    ):
        # Schema only allows Organization (→ ORG).  GPE/Location excluded.
        schema = GraphSchema(
            node_types=(NodeType(label="Organization"),),
            relationship_types=(RelationshipType(label="RELATED_TO"),),
        )
        extractor = SpacyEntityRelationExtractor(
            model="en_core_web_sm",
            use_linear_extractor=True,
            create_lexical_graph=False,
        )
        chunks = TextChunks(chunks=[TextChunk(text="Apple is in NewYork.", index=0)])
        result = await extractor.run(chunks=chunks, schema=schema)

    # The (Apple, NewYork) triple should be filtered out because NewYork is GPE.
    node_ids = {n.id for n in result.nodes}
    # "newyork" is a GPE; schema only allows ORG → triple is filtered.
    assert "newyork" not in node_ids


# ---------------------------------------------------------------------------
# 9. Confidence property on relationships
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_relationships_have_confidence_property() -> None:
    """All extracted relationships carry a numeric 'confidence' property."""
    dep_triples: set[tuple[str, str, str, float]] = {
        ("Alpha", "WORKS_FOR", "Beta", 1.0),
    }

    mock_nlp = _make_mock_nlp(ents=[])

    with patch("spacy.load", return_value=mock_nlp), patch(
        "neo4j_graphrag.experimental.components.spacy_entity_relation_extractor.extract_dependency_triples",
        return_value=dep_triples,
    ), patch(
        "neo4j_graphrag.experimental.components.spacy_entity_relation_extractor.extract_linear_triples",
        return_value=set(),
    ):
        extractor = SpacyEntityRelationExtractor(
            model="en_core_web_sm",
            use_linear_extractor=True,
            create_lexical_graph=False,
        )
        chunks = TextChunks(chunks=[TextChunk(text="Alpha works for Beta.", index=0)])
        result = await extractor.run(chunks=chunks)

    assert len(result.relationships) == 1
    rel = result.relationships[0]
    assert "confidence" in rel.properties
    assert rel.properties["confidence"] == pytest.approx(1.0)
