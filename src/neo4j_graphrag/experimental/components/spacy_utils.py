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
"""Shared utility helpers for SpaCy-based components.

This module is intentionally kept free of heavy SpaCy imports at module level
so that it can be imported safely even when SpaCy is not installed.  Functions
that accept ``spacy.tokens.Doc`` objects use ``TYPE_CHECKING`` guards so that
type checkers still see the correct annotations without causing an
``ImportError`` at runtime.
"""

from __future__ import annotations

import re
import unicodedata
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import spacy.tokens

# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

_WHITESPACE_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Lowercase *text*, apply NFC Unicode normalisation, and collapse whitespace.

    Steps applied in order:

    1. NFC Unicode normalisation — ensures that visually identical characters
       with different code-point sequences compare equal (e.g. ``"caf\\u00e9"``
       and ``"cafe\\u0301"`` both become ``"café"`` before lowercasing).
    2. Lowercase via :meth:`str.lower`.
    3. Collapse all runs of whitespace (spaces, tabs, newlines …) to a single
       ASCII space.
    4. Strip leading and trailing whitespace.

    Examples::

        >>> normalize("  Supply Chain  ")
        'supply chain'
        >>> normalize("Hello   World")
        'hello world'
        >>> normalize("")
        ''
    """
    nfc = unicodedata.normalize("NFC", text)
    return _WHITESPACE_RE.sub(" ", nfc.lower()).strip()


# ---------------------------------------------------------------------------
# Dependency-based triple extraction
# ---------------------------------------------------------------------------

#: Dependency labels that mark nominal subjects (active voice).
_SUBJ_DEPS: frozenset[str] = frozenset({"nsubj"})

#: Dependency labels that mark nominal subjects in passive constructions.
_SUBJ_PASS_DEPS: frozenset[str] = frozenset({"nsubjpass"})

#: Dependency labels that mark direct objects.
_OBJ_DEPS: frozenset[str] = frozenset({"dobj", "obj"})


def extract_dependency_triples(
    doc: "spacy.tokens.Doc",
) -> set[tuple[str, str, str, float]]:
    """Walk SpaCy's dependency tree and extract (head, relation, tail, conf) triples.

    A copy of *doc* is made before any retokenization so that the caller's
    original ``Doc`` object is never mutated.  Multi-token noun chunks and named
    entities are merged using the spaCy 3.x retokenizer API
    (``Doc.retokenize()``) so that spans such as *"Supply Chain Management"*
    become a single token before extraction.

    Extraction logic per sentence
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    For every token whose dependency label is ``ROOT`` (the main verb):

    1. Collect all ``nsubj`` children → *active subjects*.
    2. Collect all ``nsubjpass`` children → *passive subjects*.
    3. Collect all ``dobj`` / ``obj`` children → *direct objects*.
    4. For each ``(subject, object)`` pair emit a primary triple:

       - **Active**: ``(subject_text, verb_lemma, object_text, conf)``
       - **Passive**: ``(object_text, verb_lemma_BY, subject_text, conf)``
         where ``verb_lemma_BY = verb_lemma.upper() + "_BY"``.

    5. For each ``prep`` child of the root verb, collect its ``pobj`` children
       and emit a secondary triple:

       - When a direct object is present:
         ``(direct_object_text, prep_label.upper(), pobj_text, 0.7)``
       - When no direct object is present:
         ``(subject_text, prep_label.upper(), pobj_text, 0.7)``

    Confidence scoring
    ~~~~~~~~~~~~~~~~~~
    * Active SVO triples: ``1.0``
    * Passive triples: ``0.9``
    * Prepositional triples: ``0.7``

    Parameters
    ----------
    doc:
        A SpaCy ``Doc`` object.  The function always works on an internal copy;
        the caller's doc is never modified.

    Returns
    -------
    set[tuple[str, str, str, float]]
        A set of ``(head, relation, tail, confidence)`` 4-tuples.  Both *head*
        and *tail* are token texts after retokenization (not normalised —
        normalisation is the caller's responsibility).
    """
    # Apply phrasal merging so that multi-word NPs become single tokens.
    # We use the spaCy 3.x retokenizer API (Span.merge() was removed in 3.0).
    # Work on a copy to avoid mutating the caller's doc.
    try:
        import spacy.tokens  # noqa: F401 — ensure spacy is available

        doc = doc.copy()

        # Merge noun chunks (requires DEP annotation for sentence boundaries).
        if doc.has_annotation("DEP"):
            try:
                with doc.retokenize() as retokenizer:
                    for span in list(doc.noun_chunks):
                        retokenizer.merge(span)
            except Exception:
                pass  # noun-chunk merging is best-effort on a blank model

        # Merge named entities (works whenever .ents is populated).
        if doc.ents:
            try:
                with doc.retokenize() as retokenizer:
                    for span in list(doc.ents):
                        retokenizer.merge(span)
            except Exception:
                pass  # overlapping or already-merged spans; skip

    except ImportError:
        pass  # SpaCy not installed — proceed without merging

    triples: set[tuple[str, str, str, float]] = set()

    # Iterate sentences; fall back to treating the whole doc as one sentence
    # if sentence boundaries are not available (no DEP annotation, no sentencizer).
    try:
        sents_iter = list(doc.sents)
    except ValueError:
        sents_iter = [doc[:]]

    for sent in sents_iter:
        for token in sent:
            if token.dep_ != "ROOT":
                continue

            verb_lemma = (token.lemma_ or token.text).upper()

            # Collect subject tokens.
            active_subjects = [c for c in token.children if c.dep_ in _SUBJ_DEPS]
            passive_subjects = [c for c in token.children if c.dep_ in _SUBJ_PASS_DEPS]
            direct_objects = [c for c in token.children if c.dep_ in _OBJ_DEPS]

            # Prepositional phrases hanging off the root verb.
            prep_children = [c for c in token.children if c.dep_ == "prep"]

            # --- Active SVO triples -------------------------------------------
            for subj in active_subjects:
                for obj in direct_objects:
                    triples.add((subj.text, verb_lemma, obj.text, 1.0))

                    # Secondary prep triples: (obj, PREP_LABEL, pobj)
                    for prep in prep_children:
                        for pobj in prep.children:
                            if pobj.dep_ == "pobj":
                                triples.add(
                                    (obj.text, prep.text.upper(), pobj.text, 0.7)
                                )

            # --- Passive voice triples ----------------------------------------
            # In passive, the grammatical subject is the logical object, and the
            # prepositional "by" phrase (dep_ == "agent") contains the logical
            # subject.  We handle two cases:
            #
            # Case A: nsubjpass + dobj (uncommon ditransitive passive)
            # Case B: nsubjpass alone — the "agent" (by-phrase) child provides
            #         the logical actor.

            for subj in passive_subjects:
                passive_relation = verb_lemma + "_BY"

                # Case A: direct object still present (ditransitive passive)
                for obj in direct_objects:
                    triples.add((obj.text, passive_relation, subj.text, 0.9))

                # Case B: "by" agent provides the logical subject
                for agent in (c for c in token.children if c.dep_ == "agent"):
                    for pobj in agent.children:
                        if pobj.dep_ == "pobj":
                            triples.add((subj.text, passive_relation, pobj.text, 0.9))

                # Secondary prep triples for the passive subject
                for prep in prep_children:
                    for pobj in prep.children:
                        if pobj.dep_ == "pobj":
                            triples.add((subj.text, prep.text.upper(), pobj.text, 0.7))

            # --- Subject-verb-only (no dobj): preps hang off verb itself ------
            # When there is no direct object, emit (subject, PREP_LABEL, pobj).
            if not direct_objects:
                for subj in active_subjects + passive_subjects:
                    for prep in prep_children:
                        for pobj in prep.children:
                            if pobj.dep_ == "pobj":
                                triples.add(
                                    (subj.text, prep.text.upper(), pobj.text, 0.7)
                                )

    return triples


# ---------------------------------------------------------------------------
# Stop-word filter
# ---------------------------------------------------------------------------

#: Common English stopwords that are excluded when normalising entity names
#: used as positions / keys in knowledge-graph nodes.  Keeping the set small
#: and stable avoids accidentally filtering out legitimate entity fragments.
WORD_FILTER: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "shall",
        "can",
        "of",
        "in",
        "to",
        "for",
        "on",
        "at",
        "by",
        "from",
        "with",
        "and",
        "or",
        "but",
        "not",
        "no",
        "nor",
        "so",
        "yet",
        "both",
        "either",
        "neither",
        "each",
        "this",
        "that",
        "these",
        "those",
        "i",
        "me",
        "my",
        "we",
        "our",
        "you",
        "your",
        "he",
        "him",
        "his",
        "she",
        "her",
        "it",
        "its",
        "they",
        "them",
        "their",
        "what",
        "which",
        "who",
        "whom",
        "whose",
        "when",
        "where",
        "why",
        "how",
        "all",
        "any",
        "some",
        "more",
        "most",
        "other",
        "such",
        "than",
        "then",
        "as",
        "if",
        "about",
        "up",
        "out",
        "into",
        "over",
        "after",
    }
)
