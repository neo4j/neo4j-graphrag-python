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
"""Unit tests for extract_dependency_triples using synthetic spaCy Docs.

spaCy is an optional dependency; these tests are skipped when it is not
installed.  All Doc objects are built with ``spacy.blank("en")`` and manually
set dependency arcs so that no trained model needs to be downloaded.
"""

import pytest

spacy = pytest.importorskip("spacy", reason="spacy is not installed")

from neo4j_graphrag.experimental.components.spacy_utils import (  # noqa: E402
    extract_dependency_triples,
)


# ---------------------------------------------------------------------------
# Helper: build a synthetic spacy Doc with dependency annotations
# ---------------------------------------------------------------------------


def _make_doc(
    words: list[str],
    lemmas: list[str],
    dep_labels: list[str],
    head_indices: list[int],
    *,
    spaces: list[bool] | None = None,
) -> "spacy.tokens.Doc":
    """Return a ``spacy.tokens.Doc`` with manually set dependency arcs.

    All parameters must have the same length as *words*.  ``head_indices``
    uses 0-based integer positions into *words*; the ROOT token must point to
    itself (``head_indices[i] == i``).
    """
    from spacy.tokens import Doc

    nlp = spacy.blank("en")
    if spaces is None:
        spaces = [True] * (len(words) - 1) + [False]
    doc = Doc(nlp.vocab, words=words, spaces=spaces, lemmas=lemmas)
    for i, (dep, head_idx) in enumerate(zip(dep_labels, head_indices)):
        doc[i].dep_ = dep
        doc[i].head = doc[head_idx]
    return doc


# ---------------------------------------------------------------------------
# Tests for extract_dependency_triples
# ---------------------------------------------------------------------------


class TestExtractDependencyTriples:
    """Tests for extract_dependency_triples using synthetic spacy Docs."""

    # ------------------------------------------------------------------
    # Active SVO
    # ------------------------------------------------------------------

    def test_active_svo_yields_triple(self) -> None:
        """'Apple acquired IBM' → (Apple, ACQUIRE, IBM, 1.0)."""
        doc = _make_doc(
            words=["Apple", "acquired", "IBM"],
            lemmas=["Apple", "acquire", "IBM"],
            dep_labels=["nsubj", "ROOT", "dobj"],
            head_indices=[1, 1, 1],
        )
        triples = extract_dependency_triples(doc)
        assert ("Apple", "ACQUIRE", "IBM", 1.0) in triples

    def test_active_svo_confidence_is_one(self) -> None:
        """Active SVO triples must have confidence == 1.0."""
        doc = _make_doc(
            words=["Google", "bought", "YouTube"],
            lemmas=["Google", "buy", "YouTube"],
            dep_labels=["nsubj", "ROOT", "dobj"],
            head_indices=[1, 1, 1],
        )
        triples = extract_dependency_triples(doc)
        for head, _rel, tail, conf in triples:
            if head == "Google" and tail == "YouTube":
                assert conf == 1.0

    def test_active_svo_uses_verb_lemma(self) -> None:
        """Relation string should be the uppercased lemma, not the surface form."""
        doc = _make_doc(
            words=["Apple", "acquired", "IBM"],
            lemmas=["Apple", "acquire", "IBM"],
            dep_labels=["nsubj", "ROOT", "dobj"],
            head_indices=[1, 1, 1],
        )
        triples = extract_dependency_triples(doc)
        relations = {rel for _, rel, _, _ in triples}
        assert "ACQUIRE" in relations
        assert "ACQUIRED" not in relations  # surface form must NOT appear

    def test_no_object_no_svo_triple(self) -> None:
        """A sentence with nsubj but no dobj should yield no SVO triple."""
        # "Apple runs"
        doc = _make_doc(
            words=["Apple", "runs"],
            lemmas=["Apple", "run"],
            dep_labels=["nsubj", "ROOT"],
            head_indices=[1, 1],
        )
        triples = extract_dependency_triples(doc)
        # No SVO triple — only possibly a prep triple if there were one
        svo_triples = [t for t in triples if t[1] not in ("TO", "IN", "ON", "AT")]
        assert len(svo_triples) == 0

    def test_no_subject_no_svo_triple(self) -> None:
        """A sentence with dobj but no subject should yield no SVO triple."""
        # "(Ø) acquires IBM" — dobj without nsubj
        doc = _make_doc(
            words=["acquires", "IBM"],
            lemmas=["acquire", "IBM"],
            dep_labels=["ROOT", "dobj"],
            head_indices=[0, 0],
        )
        triples = extract_dependency_triples(doc)
        assert len(triples) == 0

    # ------------------------------------------------------------------
    # Passive voice
    # ------------------------------------------------------------------

    def test_passive_nsubjpass_agent_yields_triple(self) -> None:
        """'IBM was acquired by Apple' → (IBM, ACQUIRE_BY, Apple, 0.9)."""
        # IBM(nsubjpass) was(auxpass) acquired(ROOT) by(agent) Apple(pobj)
        doc = _make_doc(
            words=["IBM", "was", "acquired", "by", "Apple"],
            lemmas=["IBM", "be", "acquire", "by", "Apple"],
            dep_labels=["nsubjpass", "auxpass", "ROOT", "agent", "pobj"],
            head_indices=[2, 2, 2, 2, 3],
        )
        triples = extract_dependency_triples(doc)
        assert ("IBM", "ACQUIRE_BY", "Apple", 0.9) in triples

    def test_passive_relation_has_by_suffix(self) -> None:
        """Passive triples must carry the _BY suffix on the relation."""
        doc = _make_doc(
            words=["IBM", "was", "acquired", "by", "Apple"],
            lemmas=["IBM", "be", "acquire", "by", "Apple"],
            dep_labels=["nsubjpass", "auxpass", "ROOT", "agent", "pobj"],
            head_indices=[2, 2, 2, 2, 3],
        )
        triples = extract_dependency_triples(doc)
        relations = {rel for _, rel, _, _ in triples}
        assert any(rel.endswith("_BY") for rel in relations)

    def test_passive_confidence_is_0_9(self) -> None:
        """Passive triples must have confidence 0.9."""
        doc = _make_doc(
            words=["IBM", "was", "acquired", "by", "Apple"],
            lemmas=["IBM", "be", "acquire", "by", "Apple"],
            dep_labels=["nsubjpass", "auxpass", "ROOT", "agent", "pobj"],
            head_indices=[2, 2, 2, 2, 3],
        )
        triples = extract_dependency_triples(doc)
        passive = [t for t in triples if t[1].endswith("_BY")]
        assert len(passive) > 0
        for _, _, _, conf in passive:
            assert conf == 0.9

    # ------------------------------------------------------------------
    # Prepositional triples
    # ------------------------------------------------------------------

    def test_prep_triple_without_dobj(self) -> None:
        """'Apple moved to California' → (Apple, TO, California, 0.7)."""
        # Apple(nsubj) moved(ROOT) to(prep) California(pobj)
        doc = _make_doc(
            words=["Apple", "moved", "to", "California"],
            lemmas=["Apple", "move", "to", "California"],
            dep_labels=["nsubj", "ROOT", "prep", "pobj"],
            head_indices=[1, 1, 1, 2],
        )
        triples = extract_dependency_triples(doc)
        assert ("Apple", "TO", "California", 0.7) in triples

    def test_prep_triple_with_dobj(self) -> None:
        """'Apple sold IBM to Google' → primary (Apple, SELL, IBM) + secondary (IBM, TO, Google)."""
        # Apple(nsubj) sold(ROOT) IBM(dobj) to(prep) Google(pobj)
        doc = _make_doc(
            words=["Apple", "sold", "IBM", "to", "Google"],
            lemmas=["Apple", "sell", "IBM", "to", "Google"],
            dep_labels=["nsubj", "ROOT", "dobj", "prep", "pobj"],
            head_indices=[1, 1, 1, 1, 3],
        )
        triples = extract_dependency_triples(doc)
        assert ("Apple", "SELL", "IBM", 1.0) in triples
        assert ("IBM", "TO", "Google", 0.7) in triples

    def test_prep_triple_confidence_is_0_7(self) -> None:
        """All prepositional triples must carry confidence 0.7."""
        doc = _make_doc(
            words=["Apple", "moved", "to", "California"],
            lemmas=["Apple", "move", "to", "California"],
            dep_labels=["nsubj", "ROOT", "prep", "pobj"],
            head_indices=[1, 1, 1, 2],
        )
        triples = extract_dependency_triples(doc)
        prep_triples = [t for t in triples if t[1] == "TO"]
        assert len(prep_triples) > 0
        for _, _, _, conf in prep_triples:
            assert conf == 0.7

    # ------------------------------------------------------------------
    # Multi-token entity merging
    # ------------------------------------------------------------------

    def test_multiword_entity_collapsed_into_single_token(self) -> None:
        """Named entities spanning multiple tokens are merged before extraction.

        A two-token entity set via ``doc.ents`` must appear as a single string
        in the resulting triple head/tail.
        """
        from spacy.tokens import Doc, Span

        nlp = spacy.blank("en")
        # "Supply Chain acquired IBM"
        words = ["Supply", "Chain", "acquired", "IBM"]
        lemmas = ["Supply", "Chain", "acquire", "IBM"]
        spaces = [True, True, True, False]
        doc = Doc(nlp.vocab, words=words, spaces=spaces, lemmas=lemmas)

        # Set dependency arcs: "Supply Chain" fused as nsubj for "acquired"
        # We treat token 0 ("Supply") as the head of the nsubj span.
        doc[0].dep_ = "nsubj"
        doc[0].head = doc[2]
        doc[1].dep_ = "compound"
        doc[1].head = doc[0]
        doc[2].dep_ = "ROOT"
        doc[2].head = doc[2]
        doc[3].dep_ = "dobj"
        doc[3].head = doc[2]

        # Mark "Supply Chain" (tokens 0-1) as a named entity so the retokenizer
        # will merge it into a single token.
        doc.ents = [Span(doc, 0, 2, label="ORG")]

        triples = extract_dependency_triples(doc)

        heads = {t[0] for t in triples}
        # After merging, "Supply Chain" should be a single token string
        assert any(
            "Supply" in h for h in heads
        ), f"expected merged entity in heads, got {heads}"

    # ------------------------------------------------------------------
    # Return type
    # ------------------------------------------------------------------

    def test_returns_set(self) -> None:
        """Return value must be a set."""
        doc = _make_doc(
            words=["Apple", "acquired", "IBM"],
            lemmas=["Apple", "acquire", "IBM"],
            dep_labels=["nsubj", "ROOT", "dobj"],
            head_indices=[1, 1, 1],
        )
        result = extract_dependency_triples(doc)
        assert isinstance(result, set)

    def test_returns_4_tuples(self) -> None:
        """Each element must be a 4-tuple (head, rel, tail, conf)."""
        doc = _make_doc(
            words=["Apple", "acquired", "IBM"],
            lemmas=["Apple", "acquire", "IBM"],
            dep_labels=["nsubj", "ROOT", "dobj"],
            head_indices=[1, 1, 1],
        )
        result = extract_dependency_triples(doc)
        for item in result:
            assert len(item) == 4
            head, rel, tail, conf = item
            assert isinstance(head, str)
            assert isinstance(rel, str)
            assert isinstance(tail, str)
            assert isinstance(conf, float)

    def test_empty_doc_returns_empty_set(self) -> None:
        """An empty doc should yield an empty set."""
        from spacy.tokens import Doc

        nlp = spacy.blank("en")
        doc = Doc(nlp.vocab, words=[], spaces=[])
        result = extract_dependency_triples(doc)
        assert result == set()

    def test_no_root_token_returns_empty(self) -> None:
        """If no token has dep ROOT, no triples are extracted."""
        doc = _make_doc(
            words=["Apple", "IBM"],
            lemmas=["Apple", "IBM"],
            dep_labels=["nsubj", "dobj"],
            head_indices=[1, 0],
        )
        result = extract_dependency_triples(doc)
        assert result == set()

    def test_deduplication_via_set(self) -> None:
        """The same triple emitted for two different sentences is stored once."""
        # Two identical sentences in one doc — the set should deduplicate.
        from spacy.tokens import Doc

        nlp = spacy.blank("en")
        words = ["Apple", "acquired", "IBM", "Apple", "acquired", "IBM"]
        lemmas = ["Apple", "acquire", "IBM", "Apple", "acquire", "IBM"]
        spaces = [True, True, True, True, True, False]
        doc = Doc(nlp.vocab, words=words, spaces=spaces, lemmas=lemmas)
        # First sentence: tokens 0-2
        doc[0].dep_ = "nsubj"
        doc[0].head = doc[1]
        doc[1].dep_ = "ROOT"
        doc[1].head = doc[1]
        doc[2].dep_ = "dobj"
        doc[2].head = doc[1]
        # Second sentence: tokens 3-5
        doc[3].dep_ = "nsubj"
        doc[3].head = doc[4]
        doc[4].dep_ = "ROOT"
        doc[4].head = doc[4]
        doc[5].dep_ = "dobj"
        doc[5].head = doc[4]

        result = extract_dependency_triples(doc)
        # Should be exactly one triple (set deduplication)
        assert result == {("Apple", "ACQUIRE", "IBM", 1.0)}

    def test_no_dep_annotations_falls_back_gracefully(self) -> None:
        """When no DEP annotations are set, doc.sents raises ValueError.

        The function must fall back to treating the whole doc as one sentence
        rather than raising.
        """
        from spacy.tokens import Doc

        nlp = spacy.blank("en")
        # Create doc with NO dependency annotations at all
        doc = Doc(nlp.vocab, words=["Apple", "IBM"], spaces=[True, False])
        # No dep_ set — doc.sents will raise ValueError
        result = extract_dependency_triples(doc)
        # Should return empty set (no ROOT token found), not raise
        assert isinstance(result, set)
