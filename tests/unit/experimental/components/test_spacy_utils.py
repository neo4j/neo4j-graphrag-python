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
"""Unit tests for spacy_utils.normalize() and WORD_FILTER."""

import pytest

from neo4j_graphrag.experimental.components.spacy_utils import WORD_FILTER, normalize


class TestNormalize:
    def test_basic_lowercase(self) -> None:
        assert normalize("Hello World") == "hello world"

    def test_leading_trailing_spaces(self) -> None:
        assert normalize("  Supply Chain  ") == "supply chain"

    def test_multiple_internal_spaces(self) -> None:
        assert normalize("Hello   World") == "hello world"

    def test_empty_string(self) -> None:
        assert normalize("") == ""

    def test_whitespace_only(self) -> None:
        assert normalize("   ") == ""

    def test_tabs_and_newlines(self) -> None:
        assert normalize("foo\t\nbar") == "foo bar"

    def test_already_normalized(self) -> None:
        assert normalize("supply chain") == "supply chain"

    def test_mixed_case_with_spaces(self) -> None:
        assert normalize("  UPPER lower MiXeD  ") == "upper lower mixed"

    def test_single_word(self) -> None:
        assert normalize("Apple") == "apple"

    def test_numbers_unchanged(self) -> None:
        assert normalize("  123 456  ") == "123 456"

    def test_unicode_nfc_normalization(self) -> None:
        # "café" composed (U+00E9) vs decomposed (e + combining accent U+0301)
        composed = "café"       # NFC form
        decomposed = "café"    # NFD form — visually identical
        assert normalize(composed) == normalize(decomposed)

    def test_unicode_lowercase_accent(self) -> None:
        assert normalize("Ångström") == "ångström"


class TestWordFilter:
    def test_is_frozenset(self) -> None:
        assert isinstance(WORD_FILTER, frozenset)

    def test_contains_the(self) -> None:
        assert "the" in WORD_FILTER

    def test_contains_a(self) -> None:
        assert "a" in WORD_FILTER

    def test_contains_is(self) -> None:
        assert "is" in WORD_FILTER

    def test_contains_and(self) -> None:
        assert "and" in WORD_FILTER

    def test_contains_of(self) -> None:
        assert "of" in WORD_FILTER

    def test_does_not_contain_entity_words(self) -> None:
        # Common entity-like words should not be in the filter
        for word in ("apple", "google", "london", "supply", "chain"):
            assert word not in WORD_FILTER

    def test_all_entries_are_lowercase(self) -> None:
        for word in WORD_FILTER:
            assert word == word.lower(), f"WORD_FILTER entry '{word}' is not lowercase"

    def test_immutable(self) -> None:
        with pytest.raises(AttributeError):
            WORD_FILTER.add("new_word")  # type: ignore[attr-defined]
