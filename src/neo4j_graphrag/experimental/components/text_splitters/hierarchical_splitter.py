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
"""Hierarchical text splitter that detects section boundaries before chunking.

Supported header strategies:

* ``"markdown"`` — ATX Markdown header lines (``# H1``, ``## H2``, … up to ``###### H6``): one to six ``#`` characters followed by a space and header text.
* ``"capitalization"`` — short Title Case or ALL_CAPS lines without terminal
  punctuation (appropriate for plain-text output from loaders like
  LiteParseLoader).
* ``"blank_line"`` — short lines that are surrounded by blank lines on both
  sides (a common plain-text section marker).
* ``"spacy_verbless"`` — SpaCy-parsed sentences that are short, contain no
  verb, and are followed by a longer sentence.

All strategies produce a list of *sections* (contiguous text blocks).  Each
section is then emitted as a single chunk when it fits within ``max_chunk_size``,
or recursively split with ``chunk_overlap`` when it is larger.

Optionally, when ``drop_verbless_sentences=True`` (default), SpaCy is used to
remove sentences with no verb token from every emitted chunk.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from pydantic import validate_call

from neo4j_graphrag.experimental.components.text_splitters.base import TextSplitter
from neo4j_graphrag.experimental.components.types import TextChunk, TextChunks

# ---------------------------------------------------------------------------
# Header-detection helpers
# ---------------------------------------------------------------------------

_MARKDOWN_HEADER_RE = re.compile(r"^#{1,6}\s+\S", re.MULTILINE)

# Title Case: starts with capital, no terminal punctuation at end of line,
# short (≤ 80 chars), and at least half the words are Title-cased.
_TITLECASE_RE = re.compile(r"^[A-Z][^\n]{0,78}$")
_TERMINAL_PUNCT_RE = re.compile(r"[.!?;,]$")

# ALL_CAPS line: all uppercase letters / spaces / digits
_ALLCAPS_RE = re.compile(r"^[A-Z0-9][A-Z0-9 \t\-:]+$")

# Blank-line boundary: a line that is non-empty, short, and preceded/followed
# by blank lines (handled at section-split level, not a single regex).
_SHORT_LINE_MAX = 80


def _is_title_case(line: str) -> bool:
    """Return True when a line looks like a Title Case heading."""
    if not _TITLECASE_RE.match(line):
        return False
    if _TERMINAL_PUNCT_RE.search(line):
        return False
    words = line.split()
    if len(words) == 0:
        return False
    capitalised = sum(1 for w in words if w and w[0].isupper())
    return capitalised / len(words) >= 0.5


def _is_allcaps(line: str) -> bool:
    """Return True when a line is ALL CAPS without terminal punctuation."""
    if not _ALLCAPS_RE.match(line):
        return False
    return not _TERMINAL_PUNCT_RE.search(line)


def _split_at_markdown_headers(text: str) -> list[str]:
    """Split *text* into sections at Markdown ATX header lines (``#``, ``##``, …)."""
    # We split before each header line so that the header stays with its section.
    lines = text.splitlines(keepends=True)
    sections: list[str] = []
    current: list[str] = []
    for line in lines:
        if re.match(r"^#{1,6}\s+\S", line) and current:
            sections.append("".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        sections.append("".join(current))
    return [s for s in sections if s.strip()]


def _split_at_capitalization(text: str) -> list[str]:
    """Split *text* into sections at Title Case or ALL_CAPS heading lines."""
    lines = text.splitlines(keepends=True)
    sections: list[str] = []
    current: list[str] = []
    for line in lines:
        stripped = line.rstrip("\r\n")
        is_header = (
            len(stripped) <= _SHORT_LINE_MAX
            and stripped.strip()
            and (_is_title_case(stripped.strip()) or _is_allcaps(stripped.strip()))
        )
        if is_header and current:
            sections.append("".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        sections.append("".join(current))
    return [s for s in sections if s.strip()]


def _split_at_blank_line(text: str) -> list[str]:
    """Split *text* into sections at short lines surrounded by blank lines.

    A line qualifies as a section header when it:

    * Is non-empty and at most 80 characters.
    * Contains no terminal punctuation (not a regular sentence).
    * Has no more than 6 words (avoids treating body sentences as headers).
    * Is preceded by a blank line (or is the very first non-blank line).
    * Is followed by a blank line.
    """
    lines = text.splitlines(keepends=True)
    n = len(lines)
    sections: list[str] = []
    current: list[str] = []

    i = 0
    while i < n:
        line = lines[i]
        stripped = line.rstrip("\r\n").strip()
        # A "blank-line boundary" header: non-empty, short, preceded by blank,
        # followed by blank, few words, no terminal punctuation.
        prev_blank = (i == 0) or (not lines[i - 1].strip())
        next_blank = (i + 1 >= n) or (not lines[i + 1].strip())
        word_count = len(stripped.split()) if stripped else 0
        is_header = (
            stripped
            and len(stripped) <= _SHORT_LINE_MAX
            and not _TERMINAL_PUNCT_RE.search(stripped)
            and word_count <= 6
            and prev_blank
            and next_blank
        )
        if is_header and current:
            sections.append("".join(current))
            current = [line]
        else:
            current.append(line)
        i += 1

    if current:
        sections.append("".join(current))
    return [s for s in sections if s.strip()]


def _split_at_spacy_verbless(text: str, nlp: Any) -> list[str]:
    """Split *text* into sections at SpaCy-detected verbless heading sentences.

    A sentence qualifies as a heading when it:

    * Has at most 80 characters.
    * Contains no verb token (POS tag ``VERB`` or ``AUX``).
    * Is immediately followed by a longer sentence (> 80 chars).
    """
    doc = nlp(text)
    sentences = list(doc.sents)
    if not sentences:
        return [text] if text.strip() else []

    # Identify which sentences are "headers".
    is_header = [False] * len(sentences)
    for idx, sent in enumerate(sentences):
        sent_text = sent.text.strip()
        if len(sent_text) > _SHORT_LINE_MAX:
            continue
        has_verb = any(tok.pos_ in ("VERB", "AUX") for tok in sent)
        if has_verb:
            continue
        # Must be followed by a longer sentence.
        if (
            idx + 1 < len(sentences)
            and len(sentences[idx + 1].text.strip()) > _SHORT_LINE_MAX
        ):
            is_header[idx] = True

    # Build sections: split before each header sentence (except the first).
    sections: list[str] = []
    current_parts: list[str] = []
    for idx, sent in enumerate(sentences):
        if is_header[idx] and current_parts:
            sections.append(" ".join(current_parts))
            current_parts = [sent.text]
        else:
            current_parts.append(sent.text)

    if current_parts:
        sections.append(" ".join(current_parts))

    return [s for s in sections if s.strip()]


# ---------------------------------------------------------------------------
# Overlap-based character splitter (for sections larger than max_chunk_size)
# ---------------------------------------------------------------------------


def _split_with_overlap(text: str, max_size: int, overlap: int) -> list[str]:
    """Split *text* into character-level chunks of at most *max_size* with
    *overlap* characters carried over from the previous chunk.

    Splits are attempted at whitespace boundaries to avoid cutting words.
    """
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    length = len(text)
    step = max(1, max_size - overlap)

    while start < length:
        end = min(start + max_size, length)
        chunk = text[start:end]

        # Prefer to cut at a whitespace boundary when not at end of text.
        if end < length:
            # Walk backwards to find a space.
            cut = end
            while cut > start and not text[cut - 1].isspace():
                cut -= 1
            if cut > start:
                end = cut
                chunk = text[start:end]

        chunks.append(chunk)

        # Advance by step, ensuring we always make progress.
        next_start = start + step
        if next_start <= start:
            next_start = start + 1
        start = next_start

    return chunks


# ---------------------------------------------------------------------------
# Verb-filter using SpaCy
# ---------------------------------------------------------------------------


def _drop_verbless_sentences(text: str, nlp: Any) -> str:
    """Remove sentences with no verb token from *text* using SpaCy.

    A sentence is considered *verbless* when it contains no token whose
    part-of-speech tag is ``VERB`` or ``AUX``.
    """
    doc = nlp(text)
    kept: list[str] = []
    for sent in doc.sents:
        has_verb = any(tok.pos_ in ("VERB", "AUX") for tok in sent)
        if has_verb:
            kept.append(sent.text)
    return " ".join(kept)


# ---------------------------------------------------------------------------
# Valid strategy names
# ---------------------------------------------------------------------------

_VALID_STRATEGIES = frozenset(
    {"markdown", "capitalization", "blank_line", "spacy_verbless"}
)


# ---------------------------------------------------------------------------
# Main component
# ---------------------------------------------------------------------------


class HierarchicalTextSplitter(TextSplitter):
    """Splits text by first detecting section boundaries then chunking each section.

    Args:
        max_chunk_size (int): Maximum number of characters per output chunk.
            Defaults to 2048.
        chunk_overlap (int): Characters of overlap between consecutive chunks
            when a section must be further split.  Must be less than
            ``max_chunk_size``.  Defaults to 200.
        header_strategy (str): How to detect section boundaries.  One of:

            * ``"markdown"`` — Markdown ATX header lines (``#``, ``##``, …).
            * ``"capitalization"`` — short Title Case or ALL_CAPS lines without
              terminal punctuation.
            * ``"blank_line"`` — short lines surrounded by blank lines on both
              sides.
            * ``"spacy_verbless"`` — SpaCy-detected short verbless sentences
              that precede a longer sentence.

        model (str): SpaCy model name loaded when *header_strategy* is
            ``"spacy_verbless"`` or *drop_verbless_sentences* is ``True``.
            Defaults to ``"en_core_web_sm"``.
        drop_verbless_sentences (bool): When ``True`` (default), SpaCy is used
            to remove verbless sentences from every emitted chunk.  Note that
            this default value causes SpaCy to be loaded at construction time
            regardless of the chosen *header_strategy* — install
            ``neo4j-graphrag[nlp]`` when using the default, or explicitly pass
            ``drop_verbless_sentences=False`` to avoid the SpaCy dependency.

    Example:

    .. code-block:: python

        from neo4j_graphrag.experimental.components.text_splitters.hierarchical_splitter import (
            HierarchicalTextSplitter,
        )
        from neo4j_graphrag.experimental.pipeline import Pipeline

        pipeline = Pipeline()
        splitter = HierarchicalTextSplitter(
            max_chunk_size=2048,
            chunk_overlap=200,
            header_strategy="markdown",
        )
        pipeline.add_component(splitter, "text_splitter")
    """

    @validate_call
    def __init__(
        self,
        max_chunk_size: int = 2048,
        chunk_overlap: int = 200,
        header_strategy: str = "markdown",
        model: str = "en_core_web_sm",
        drop_verbless_sentences: bool = True,
    ) -> None:
        if max_chunk_size <= 0:
            raise ValueError("max_chunk_size must be strictly greater than 0")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be >= 0")
        if chunk_overlap >= max_chunk_size:
            raise ValueError("chunk_overlap must be strictly less than max_chunk_size")
        if header_strategy not in _VALID_STRATEGIES:
            raise ValueError(
                f"header_strategy must be one of {sorted(_VALID_STRATEGIES)}, "
                f"got {header_strategy!r}"
            )

        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
        self.header_strategy = header_strategy
        self.model = model
        self.drop_verbless_sentences = drop_verbless_sentences

        # Pre-load SpaCy only when needed.
        self._nlp: Optional[Any] = None
        needs_spacy = header_strategy == "spacy_verbless" or drop_verbless_sentences
        if needs_spacy:
            self._nlp = self._load_spacy(model)

    @staticmethod
    def _load_spacy(model: str) -> Any:
        """Load a SpaCy model, raising a clear error when SpaCy is missing."""
        try:
            import spacy  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "SpaCy is required for this configuration of HierarchicalTextSplitter. "
                "Install it with: pip install 'neo4j-graphrag[nlp]'"
            ) from exc
        try:
            return spacy.load(model)
        except OSError as exc:
            raise ValueError(
                f"SpaCy model {model!r} is not installed. "
                f"Download it with: python -m spacy download {model}"
            ) from exc

    def _detect_sections(self, text: str) -> list[str]:
        """Detect section boundaries and return a list of section strings."""
        strategy = self.header_strategy
        if strategy == "markdown":
            sections = _split_at_markdown_headers(text)
        elif strategy == "capitalization":
            sections = _split_at_capitalization(text)
        elif strategy == "blank_line":
            sections = _split_at_blank_line(text)
        else:  # "spacy_verbless"
            if self._nlp is None:
                raise RuntimeError(
                    "SpaCy model not loaded for 'spacy_verbless' strategy; this is a bug"
                )
            sections = _split_at_spacy_verbless(text, self._nlp)

        # Fallback: if no sections were detected, treat the whole text as one.
        if not sections:
            sections = [text] if text.strip() else []
        return sections

    def _chunk_section(self, section_text: str) -> list[str]:
        """Return one or more raw text chunks for a single *section_text*."""
        if len(section_text) <= self.max_chunk_size:
            return [section_text]
        return _split_with_overlap(
            section_text, self.max_chunk_size, self.chunk_overlap
        )

    def _filter_verbless(self, text: str) -> str:
        """Apply the verbless-sentence filter if enabled."""
        if not self.drop_verbless_sentences or self._nlp is None:
            return text
        filtered = _drop_verbless_sentences(text, self._nlp)
        # Fall back to original text when filtering removes everything.
        return filtered if filtered.strip() else text

    @validate_call
    async def run(self, text: str) -> TextChunks:
        """Split *text* into hierarchical chunks.

        Args:
            text (str): The text to be split.

        Returns:
            TextChunks: A list of chunks with sequential index values starting
            from 0.
        """
        if not text.strip():
            return TextChunks(chunks=[])

        sections = self._detect_sections(text)
        raw_chunks: list[str] = []
        for section in sections:
            raw_chunks.extend(self._chunk_section(section))

        chunks: list[TextChunk] = []
        for raw in raw_chunks:
            filtered = self._filter_verbless(raw)
            if filtered.strip():
                chunks.append(TextChunk(text=filtered, index=len(chunks)))

        return TextChunks(chunks=chunks)
