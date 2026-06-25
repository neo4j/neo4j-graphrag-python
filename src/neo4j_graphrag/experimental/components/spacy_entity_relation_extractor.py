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
"""SpaCy-based entity and relation extractor.

Uses SpaCy's NLP pipeline to extract entities and relationships from text
chunks without requiring an LLM. Extraction combines dependency-based
triple extraction and optional proximity-based linear triple extraction.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

from neo4j_graphrag.experimental.components.entity_relation_extractor import (
    EntityRelationExtractor,
    OnError,
)
from neo4j_graphrag.experimental.components.lexical_graph import LexicalGraphBuilder
from neo4j_graphrag.experimental.components.schema import GraphSchema
from neo4j_graphrag.experimental.components.spacy_linear_extractor import (
    extract_linear_triples,
)
from neo4j_graphrag.experimental.components.spacy_utils import (
    WORD_FILTER,
    extract_dependency_triples,
    normalize,
)
from neo4j_graphrag.experimental.components.types import (
    DocumentInfo,
    LexicalGraphConfig,
    Neo4jGraph,
    Neo4jNode,
    Neo4jRelationship,
    TextChunk,
    TextChunks,
)

if TYPE_CHECKING:
    import spacy
    import spacy.tokens

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SpaCy NER label → schema node-type label mapping
# ---------------------------------------------------------------------------
# Maps SpaCy's built-in NER entity type labels to commonly used schema labels.
# Used when a GraphSchema is provided to filter out unrecognised entity types.
# The mapping is intentionally broad; a case-insensitive label-match on the
# schema side covers the rest (see _build_allowed_ner_labels).

_SPACY_NER_TO_SCHEMA_LABEL: dict[str, str] = {
    "ORG": "Organization",
    "PERSON": "Person",
    "GPE": "Location",
    "LOC": "Location",
    "FAC": "Location",
    "NORP": "Group",
    "PRODUCT": "Product",
    "EVENT": "Event",
    "WORK_OF_ART": "WorkOfArt",
    "LAW": "Law",
    "LANGUAGE": "Language",
    "DATE": "Date",
    "TIME": "Time",
    "MONEY": "Money",
    "QUANTITY": "Quantity",
    "CARDINAL": "Cardinal",
    "ORDINAL": "Ordinal",
    "PERCENT": "Percent",
}


def _build_allowed_ner_labels(schema: GraphSchema) -> Optional[set[str]]:
    """Return the set of SpaCy NER labels that match schema node_type labels.

    Returns ``None`` when the schema has no node_types (no filtering).  Also
    returns ``None`` — with a warning — when node_types are defined but none
    can be mapped to a SpaCy NER label (e.g. purely custom types like
    "Contract").  This "fail-open" behaviour avoids silently filtering out all
    entities when the schema uses domain-specific labels.

    The matching strategy (in priority order):

    1. Direct case-insensitive match of the schema label against the values in
       :data:`_SPACY_NER_TO_SCHEMA_LABEL` (e.g. "Organization" → ``ORG``).
    2. Try the schema label uppercased directly as a SpaCy NER label (covers
       schemas that already use ``ORG``, ``PERSON``, ``GPE``, etc.).
    """
    if not schema.node_types:
        return None

    allowed: set[str] = set()
    schema_labels_lower = {nt.label.lower() for nt in schema.node_types}

    # Strategy 1: forward lookup in the canonical NER→schema-label map.
    for spacy_ner, schema_default in _SPACY_NER_TO_SCHEMA_LABEL.items():
        if schema_default.lower() in schema_labels_lower:
            allowed.add(spacy_ner)

    # Strategy 2: treat the schema label as a raw SpaCy NER label (e.g. "ORG").
    for label_lower in schema_labels_lower:
        upper = label_lower.upper()
        if upper in _SPACY_NER_TO_SCHEMA_LABEL:
            allowed.add(upper)

    if not allowed:
        logger.warning(
            "GraphSchema has node_types but none could be mapped to a SpaCy NER label. "
            "NER filtering will be skipped (all entity types are kept). "
            "Consider using standard schema labels such as 'Organization', 'Person', or 'Location'."
        )
        return None

    return allowed


def _build_allowed_relation_types(schema: GraphSchema) -> Optional[set[str]]:
    """Return the set of normalised relation type strings allowed by the schema.

    Each relationship_type label is uppercased and has spaces replaced by
    underscores to match the format produced by the triple extractors.
    Returns ``None`` when the schema defines no relationship_types (no filtering).
    """
    if not schema.relationship_types:
        return None
    return {rt.label.upper().replace(" ", "_") for rt in schema.relationship_types}


class SpacyEntityRelationExtractor(EntityRelationExtractor):
    """Extract entities and relations from text using SpaCy NLP pipelines.

    This extractor does not require a large language model.  It uses SpaCy's
    dependency parser and/or named-entity recogniser to produce (head, relation,
    tail) triples which are then converted to :class:`Neo4jNode` and
    :class:`Neo4jRelationship` objects.

    Args:
        model (str): SpaCy model name to load (e.g. ``"en_core_web_sm"``).
            The model is loaded once at construction time.  A
            :class:`ValueError` is raised with install instructions if the
            model is not available.
        word_filter (Optional[list[str]]): Custom list of stopwords to exclude
            when checking whether a normalised entity name is meaningful.  When
            ``None`` the built-in :data:`WORD_FILTER` constant is used.
        use_linear_extractor (bool): When ``True`` (default), proximity-based
            triples from :func:`extract_linear_triples` are unioned with the
            dependency-based triples from :func:`extract_dependency_triples`.
        create_lexical_graph (bool): Inherited from
            :class:`EntityRelationExtractor`.  When ``True`` (default), chunk
            and document nodes are added to the returned graph.
        on_error (OnError): Inherited from :class:`EntityRelationExtractor`.
            Controls behaviour when a per-chunk exception occurs.

    Example::

        from neo4j_graphrag.experimental.components.spacy_entity_relation_extractor import (
            SpacyEntityRelationExtractor,
        )
        from neo4j_graphrag.experimental.components.types import TextChunks, TextChunk

        extractor = SpacyEntityRelationExtractor(model="en_core_web_sm")
        chunks = TextChunks(chunks=[TextChunk(text="Apple bought Google.", index=0)])
        import asyncio
        graph = asyncio.run(extractor.run(chunks=chunks))
    """

    def __init__(
        self,
        model: str = "en_core_web_sm",
        word_filter: Optional[list[str]] = None,
        use_linear_extractor: bool = True,
        create_lexical_graph: bool = True,
        on_error: OnError = OnError.IGNORE,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            on_error=on_error,
            create_lexical_graph=create_lexical_graph,
            **kwargs,
        )
        self.use_linear_extractor = use_linear_extractor
        self._word_filter: frozenset[str] = (
            frozenset(word_filter) if word_filter is not None else WORD_FILTER
        )

        try:
            import spacy as _spacy

            self.nlp: spacy.language.Language = _spacy.load(model)
        except (OSError, ImportError):
            raise ValueError(
                f"SpaCy model '{model}' is not installed or spacy is not available. "
                f"Install with:\n"
                f"    pip install 'neo4j-graphrag[nlp]'\n"
                f"    python -m spacy download {model}"
            ) from None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_valid_entity(self, name: str) -> bool:
        """Return True when *name* (after normalisation) passes the word filter."""
        norm = normalize(name)
        return bool(norm) and norm not in self._word_filter

    def _triples_to_graph(
        self,
        triples: set[tuple[str, str, str, float]],
        allowed_ner_labels: Optional[set[str]],
        allowed_rel_types: Optional[set[str]],
        ner_labels_by_text: dict[str, str],
    ) -> Neo4jGraph:
        """Convert a set of (head, relation, tail, conf) triples to a Neo4jGraph.

        Applies schema filters and word-filter cleaning.

        Args:
            triples: Raw triples from the extractors.
            allowed_ner_labels: SpaCy NER label whitelist (``None`` = no filter).
            allowed_rel_types: Relation type whitelist (``None`` = no filter).
            ner_labels_by_text: Mapping of entity text → SpaCy NER label for
                entities found in the current document.  Used for NER filtering.

        Returns:
            A :class:`Neo4jGraph` with deduplicated nodes and relationships.
        """
        nodes: dict[str, Neo4jNode] = {}
        relationships: list[Neo4jRelationship] = []

        for head_text, relation, tail_text, conf in triples:
            # --- Word filter -------------------------------------------
            if not self._is_valid_entity(head_text):
                continue
            if not self._is_valid_entity(tail_text):
                continue

            # --- Relation type filter ----------------------------------
            rel_type = relation.upper().replace(" ", "_")
            if allowed_rel_types is not None and rel_type not in allowed_rel_types:
                continue

            # --- NER type filter (only when schema is provided) --------
            if allowed_ner_labels is not None:
                head_ner = ner_labels_by_text.get(head_text)
                tail_ner = ner_labels_by_text.get(tail_text)
                if head_ner is not None and head_ner not in allowed_ner_labels:
                    continue
                if tail_ner is not None and tail_ner not in allowed_ner_labels:
                    continue

            # --- Build nodes -------------------------------------------
            head_id = normalize(head_text)
            tail_id = normalize(tail_text)

            if head_id not in nodes:
                nodes[head_id] = Neo4jNode(
                    id=head_id,
                    label="Entity",
                    properties={"name": head_text},
                )
            if tail_id not in nodes:
                nodes[tail_id] = Neo4jNode(
                    id=tail_id,
                    label="Entity",
                    properties={"name": tail_text},
                )

            # --- Build relationship ------------------------------------
            relationships.append(
                Neo4jRelationship(
                    start_node_id=head_id,
                    end_node_id=tail_id,
                    type=rel_type,
                    properties={"confidence": conf},
                )
            )

        return Neo4jGraph(
            nodes=list(nodes.values()),
            relationships=relationships,
        )

    def _extract_chunk(
        self,
        chunk: TextChunk,
        allowed_ner_labels: Optional[set[str]],
        allowed_rel_types: Optional[set[str]],
    ) -> Neo4jGraph:
        """Run NLP + extraction for a single chunk and return a raw graph."""
        doc: spacy.tokens.Doc = self.nlp(chunk.text)

        # Build NER label index for schema filtering.
        # ent.text matches the original span text before any retokenization
        # performed inside extract_dependency_triples (which works on a copy).
        ner_labels_by_text: dict[str, str] = {ent.text: ent.label_ for ent in doc.ents}

        # Dependency-based triples.
        triples: set[tuple[str, str, str, float]] = extract_dependency_triples(doc)

        # Union with linear (proximity) triples when requested.
        if self.use_linear_extractor:
            triples |= extract_linear_triples(doc)

        return self._triples_to_graph(
            triples,
            allowed_ner_labels,
            allowed_rel_types,
            ner_labels_by_text,
        )

    # ------------------------------------------------------------------
    # Component entry point
    # ------------------------------------------------------------------

    async def run(
        self,
        chunks: TextChunks,
        document_info: Optional[DocumentInfo] = None,
        lexical_graph_config: Optional[LexicalGraphConfig] = None,
        schema: Optional[GraphSchema] = None,
        **kwargs: Any,
    ) -> Neo4jGraph:
        """Extract entities and relations from all chunks.

        Args:
            chunks: Text chunks to process.
            document_info: Optional document metadata (used for lexical graph).
            lexical_graph_config: Optional configuration for the lexical graph.
            schema: Optional :class:`GraphSchema` to constrain extraction.
                When provided, only entities matching schema node_type labels
                (via SpaCy NER type mapping) and relations matching schema
                relationship_type labels are kept.

        Returns:
            A :class:`Neo4jGraph` containing the extracted entities and
            relations, plus lexical-graph nodes/relationships when
            ``create_lexical_graph=True``.
        """
        # --- Schema-derived filters ------------------------------------
        allowed_ner_labels: Optional[set[str]] = None
        allowed_rel_types: Optional[set[str]] = None
        if schema is not None:
            allowed_ner_labels = _build_allowed_ner_labels(schema)
            allowed_rel_types = _build_allowed_relation_types(schema)

        # --- Lexical graph setup --------------------------------------
        lexical_graph_builder: Optional[LexicalGraphBuilder] = None
        lexical_graph: Optional[Neo4jGraph] = None

        if self.create_lexical_graph:
            config = lexical_graph_config or LexicalGraphConfig()
            lexical_graph_builder = LexicalGraphBuilder(config=config)
            lexical_graph_result = await lexical_graph_builder.run(
                text_chunks=chunks, document_info=document_info
            )
            lexical_graph = lexical_graph_result.graph
        elif lexical_graph_config:
            lexical_graph_builder = LexicalGraphBuilder(config=lexical_graph_config)

        # --- Per-chunk extraction -------------------------------------
        chunk_graphs: list[Neo4jGraph] = []
        for chunk in chunks.chunks:
            try:
                chunk_graph = self._extract_chunk(
                    chunk, allowed_ner_labels, allowed_rel_types
                )
                self.update_ids(chunk_graph, chunk)
                if lexical_graph_builder:
                    await lexical_graph_builder.process_chunk_extracted_entities(
                        chunk_graph, chunk
                    )
                chunk_graphs.append(chunk_graph)
            except Exception as exc:
                if self.on_error == OnError.RAISE:
                    raise
                logger.warning(
                    f"Skipping chunk index={chunk.index} due to error: {exc}",
                    exc_info=True,
                )

        # --- Combine --------------------------------------------------
        graph = lexical_graph.model_copy(deep=True) if lexical_graph else Neo4jGraph()
        for cg in chunk_graphs:
            graph.nodes.extend(cg.nodes)
            graph.relationships.extend(cg.relationships)

        logger.debug(
            f"SpacyEntityRelationExtractor: extracted {len(graph.nodes)} nodes "
            f"and {len(graph.relationships)} relationships from {len(chunks.chunks)} chunks."
        )
        return graph
