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

This module is intentionally kept free of SpaCy imports so that it can be
imported safely even when SpaCy is not installed.
"""

from __future__ import annotations

import re
import unicodedata

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
