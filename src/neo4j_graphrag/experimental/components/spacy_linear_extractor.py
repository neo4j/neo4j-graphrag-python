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
"""Linear (proximity-based) triple extractor for SpaCy documents.

This module implements :func:`extract_linear_triples`, which emits
``(entity_a, "RELATED_TO", entity_b, 0.3)`` triples for every pair of NER
entities that:

1. Appear in the **same sentence**.
2. Are within **10 tokens** of each other (measured from the *start* token of
   each span).
3. Have **no other NER entity** between them.

This surface-level heuristic captures appositions, list items, and implicit
co-occurrence associations that the dependency-based extractor may miss.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import spacy.tokens

__all__ = ["extract_linear_triples"]

# Confidence assigned to all proximity-based triples.
_LINEAR_CONFIDENCE: float = 0.3

# Maximum token distance (inclusive) between entity span *starts* for a pair
# to be considered "close enough" to emit a triple.
_MAX_TOKEN_DISTANCE: int = 10


def extract_linear_triples(
    doc: "spacy.tokens.Doc",
) -> set[tuple[str, str, str, float]]:
    """Return proximity-based triples for co-occurring NER entity pairs.

    For each sentence in *doc* the function scans all ordered pairs of named
    entities ``(a, b)`` where ``a`` appears before ``b``.  A triple
    ``(a.text, "RELATED_TO", b.text, 0.3)`` is emitted when:

    * The distance from ``a.start`` to ``b.start`` is ≤ 10 tokens.
    * No other named entity starts between ``a.end`` and ``b.start``.

    Both conditions are evaluated on token indices within the document
    (``spacy.tokens.Span.start`` / ``.end``).

    Parameters
    ----------
    doc:
        A processed :class:`spacy.tokens.Doc`.  Named entities must have been
        populated (``doc.ents``).

    Returns
    -------
    set[tuple[str, str, str, float]]
        A set of ``(entity_a_text, relation, entity_b_text, confidence)``
        tuples.  Each pair is emitted **at most once** (the set deduplicates).

    Examples
    --------
    >>> import spacy
    >>> nlp = spacy.blank("en")
    >>> # … build a Doc with .ents set …
    >>> triples = extract_linear_triples(doc)
    >>> ("Apple", "RELATED_TO", "Google", 0.3) in triples
    True
    """
    triples: set[tuple[str, str, str, float]] = set()

    # Determine sentence boundaries.  When no sentence segmenter has been run
    # (e.g. ``spacy.blank`` without a sentencizer), ``doc.has_annotation("SENT_START")``
    # is False and iterating ``doc.sents`` raises a ValueError.  In that case we
    # treat the whole document as a single sentence.
    try:
        sentences = list(doc.sents)
    except ValueError:
        # No sentence boundary annotation — treat the whole doc as one sentence.
        sentences = [doc[:]]

    for sent in sentences:
        # Collect entities that fall within this sentence, preserving order.
        sent_ents = [
            ent for ent in doc.ents if ent.start >= sent.start and ent.end <= sent.end
        ]

        n = len(sent_ents)
        for i in range(n):
            ent_a = sent_ents[i]
            for j in range(i + 1, n):
                ent_b = sent_ents[j]

                # ── Distance check ────────────────────────────────────────────
                distance = ent_b.start - ent_a.start
                if distance > _MAX_TOKEN_DISTANCE:
                    # Entities are sorted by position; further pairs will only
                    # be farther away, so we can break the inner loop early.
                    break

                # ── Intervening-entity check ──────────────────────────────────
                # Entities between ent_a and ent_b are those at indices i+1 …
                # j-1 in the already-sorted sent_ents list.
                has_intervening = j > i + 1  # True when there is any entity in between

                if has_intervening:
                    # Skip this pair: another entity sits between them.
                    continue

                triples.add((ent_a.text, "RELATED_TO", ent_b.text, _LINEAR_CONFIDENCE))

    return triples
