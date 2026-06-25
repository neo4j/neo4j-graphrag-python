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
"""Unit tests for :func:`extract_linear_triples`.

All tests use ``spacy.blank("en")`` with manually constructed entity spans so
that no trained SpaCy model download is required.
"""

from __future__ import annotations

import pytest

spacy = pytest.importorskip("spacy", reason="spacy is not installed")

from neo4j_graphrag.experimental.components.spacy_linear_extractor import (  # noqa: E402
    extract_linear_triples,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_doc(
    words: list[str], ent_spans: list[tuple[int, int, str]]
) -> spacy.tokens.Doc:
    """Build a ``spacy.tokens.Doc`` from a word list and entity span specs.

    Parameters
    ----------
    words:
        Token strings, e.g. ``["Apple", "bought", "Google", "."]``.
    ent_spans:
        List of ``(start_token, end_token, label)`` tuples defining named
        entities (``end_token`` is exclusive).
    """
    nlp = spacy.blank("en")
    doc = spacy.tokens.Doc(nlp.vocab, words=words)
    spans = []
    for start, end, label in ent_spans:
        span = doc[start:end]
        span.label_ = label
        spans.append(span)
    doc.ents = spans  # type: ignore[assignment]
    return doc


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestExtractLinearTriplesBasic:
    """Fundamental behaviour: pairs within 10 tokens, no intervening entity."""

    def test_two_adjacent_entities_emit_triple(self) -> None:
        """Two consecutive NER entities in the same sentence yield one triple."""
        # "Apple Google ."
        words = ["Apple", "Google", "."]
        doc = make_doc(words, [(0, 1, "ORG"), (1, 2, "ORG")])
        triples = extract_linear_triples(doc)
        assert ("Apple", "RELATED_TO", "Google", 0.3) in triples

    def test_triple_confidence_is_exactly_0_3(self) -> None:
        """Confidence on all linear triples must be exactly 0.3."""
        words = ["Apple", "and", "Google", "."]
        doc = make_doc(words, [(0, 1, "ORG"), (2, 3, "ORG")])
        triples = extract_linear_triples(doc)
        for _, _, _, conf in triples:
            assert conf == pytest.approx(0.3)

    def test_entities_within_10_tokens_emit_triple(self) -> None:
        """Entities whose start tokens differ by exactly 10 yield a triple."""
        # tokens 0..11, entities at 0 and 10 → distance = 10 (inclusive)
        words = ["A"] + ["w"] * 9 + ["B", "."]
        doc = make_doc(words, [(0, 1, "ORG"), (10, 11, "ORG")])
        triples = extract_linear_triples(doc)
        assert ("A", "RELATED_TO", "B", 0.3) in triples

    def test_entities_more_than_10_tokens_apart_produce_no_triple(self) -> None:
        """Two NER entities more than 10 tokens apart produce no triple."""
        # entities at token 0 and 11 → distance = 11 > 10
        words = ["A"] + ["w"] * 10 + ["B", "."]
        doc = make_doc(words, [(0, 1, "ORG"), (11, 12, "ORG")])
        triples = extract_linear_triples(doc)
        assert len(triples) == 0

    def test_empty_doc_returns_empty_set(self) -> None:
        words = ["."]
        doc = make_doc(words, [])
        assert extract_linear_triples(doc) == set()

    def test_single_entity_returns_empty_set(self) -> None:
        words = ["Apple", "."]
        doc = make_doc(words, [(0, 1, "ORG")])
        assert extract_linear_triples(doc) == set()

    def test_multi_token_entity_start_to_start_distance(self) -> None:
        """Distance is measured start-to-start: a two-token entity at [0,2) and
        a one-token entity at [8,9) have distance 8 (≤ 10) and should yield a triple."""
        # "New York w w w w w w London ."
        # ent_a: tokens 0-1 ("New York"), ent_b: token 8 ("London")
        # distance = 8 - 0 = 8 ≤ 10  → triple emitted
        words = ["New", "York"] + ["w"] * 6 + ["London", "."]
        doc = make_doc(words, [(0, 2, "GPE"), (8, 9, "GPE")])
        triples = extract_linear_triples(doc)
        assert ("New York", "RELATED_TO", "London", 0.3) in triples


class TestExtractLinearTriplesInterveningEntity:
    """Intervening entity blocks the pair."""

    def test_three_entities_a_b_c_yields_only_adjacent_pairs(self) -> None:
        """A–B–C within 10 tokens: A-RELATED_TO-B and B-RELATED_TO-C but NOT A-RELATED_TO-C."""
        # tokens: Apple(0) bought(1) Google(2) from(3) Microsoft(4) .
        words = ["Apple", "bought", "Google", "from", "Microsoft", "."]
        doc = make_doc(words, [(0, 1, "ORG"), (2, 3, "ORG"), (4, 5, "ORG")])
        triples = extract_linear_triples(doc)

        assert ("Apple", "RELATED_TO", "Google", 0.3) in triples
        assert ("Google", "RELATED_TO", "Microsoft", 0.3) in triples
        # A-C must NOT be present because Google intervenes
        assert ("Apple", "RELATED_TO", "Microsoft", 0.3) not in triples

    def test_intervening_entity_blocks_pair(self) -> None:
        """Two entities with an intervening one are NOT directly paired."""
        words = ["A", "x", "B", "x", "C", "."]
        doc = make_doc(words, [(0, 1, "ORG"), (2, 3, "ORG"), (4, 5, "ORG")])
        triples = extract_linear_triples(doc)
        assert ("A", "RELATED_TO", "C", 0.3) not in triples


class TestExtractLinearTriplesSentenceBoundary:
    """Entities in different sentences must not be paired."""

    def test_entities_in_different_sentences_produce_no_triple(self) -> None:
        """Entities separated by a sentence boundary are not paired."""
        nlp = spacy.blank("en")
        # Two sentences: "Apple ." and "Google ."
        doc = spacy.tokens.Doc(
            nlp.vocab,
            words=["Apple", ".", "Google", "."],
            sent_starts=[True, False, True, False],
        )
        span_a = doc[0:1]
        span_a.label_ = "ORG"
        span_b = doc[2:3]
        span_b.label_ = "ORG"
        doc.ents = [span_a, span_b]

        triples = extract_linear_triples(doc)
        assert ("Apple", "RELATED_TO", "Google", 0.3) not in triples
        assert len(triples) == 0

    def test_entities_same_sentence_still_paired(self) -> None:
        """Entities in the same sentence ARE paired regardless of other sentences."""
        nlp = spacy.blank("en")
        # Sentence 1: "X ."  Sentence 2: "Apple and Google ."
        doc = spacy.tokens.Doc(
            nlp.vocab,
            words=["X", ".", "Apple", "Google", "."],
            sent_starts=[True, False, True, False, False],
        )
        span_a = doc[2:3]
        span_a.label_ = "ORG"
        span_b = doc[3:4]
        span_b.label_ = "ORG"
        doc.ents = [span_a, span_b]

        triples = extract_linear_triples(doc)
        assert ("Apple", "RELATED_TO", "Google", 0.3) in triples


class TestExtractLinearTriplesReturnType:
    """Return value is always a set of 4-tuples."""

    def test_return_type_is_set(self) -> None:
        words = ["Apple", "Google", "."]
        doc = make_doc(words, [(0, 1, "ORG"), (1, 2, "ORG")])
        result = extract_linear_triples(doc)
        assert isinstance(result, set)

    def test_tuples_have_four_elements(self) -> None:
        words = ["Apple", "Google", "."]
        doc = make_doc(words, [(0, 1, "ORG"), (1, 2, "ORG")])
        result = extract_linear_triples(doc)
        for triple in result:
            assert len(triple) == 4

    def test_deduplication(self) -> None:
        """Identical entity texts produce at most one triple in the set."""
        words = ["Apple", "Google", "."]
        doc = make_doc(words, [(0, 1, "ORG"), (1, 2, "ORG")])
        result = extract_linear_triples(doc)
        # Set semantics guarantee no duplicates by definition; count directly.
        apple_google = [t for t in result if t[0] == "Apple" and t[2] == "Google"]
        assert len(apple_google) == 1
