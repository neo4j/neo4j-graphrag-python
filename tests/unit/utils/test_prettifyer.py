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

from unittest.mock import Mock, call, patch

from neo4j_graphrag.utils.logging import Prettifier, prettify


def test_prettifier_short_str() -> None:
    p = Prettifier()
    value = "ab"
    pretty_value = p._prettify_str(value)
    assert pretty_value == "ab"


def test_prettifier_long_str() -> None:
    p = Prettifier()
    value = "ab" * 200
    pretty_value = p._prettify_str(value)
    assert pretty_value == "ab" * 100 + "... (200 chars)"


@patch("neo4j_graphrag.utils.logging.os.environ")
def test_prettifier_str_custom_max_length(mock_env: Mock) -> None:
    mock_env.return_value = {"LOGGING__MAX_STRING_LENGTH": "1"}
    p = Prettifier()
    value = "abc" * 100
    pretty_value = p._prettify_str(value)
    assert pretty_value == "a" + "... (299 chars)"


def test_prettifier_short_list() -> None:
    p = Prettifier()
    value = list("abc")
    pretty_value = p._prettify_list(value)
    assert pretty_value == ["a", "b", "c"]


def test_prettifier_long_list() -> None:
    p = Prettifier()
    value = list("abc") * 10
    pretty_value = p._prettify_list(value)
    assert pretty_value == ["a", "b", "c", "a", "b", "... (25 items)"]


@patch("neo4j_graphrag.utils.logging.os.environ")
def test_prettifier_list_custom_max_length(mock_env: Mock) -> None:
    mock_env.return_value = {"LOGGING__MAX_LIST_LENGTH": "1"}
    p = Prettifier()
    value = list("abc") * 10
    pretty_value = p._prettify_list(value)
    assert pretty_value == ["a", "... (29 items)"]


def test_prettifier_list_nested() -> None:
    with patch.object(
        Prettifier, "_prettify_str", return_value="mocked string"
    ) as mock:
        p = Prettifier()
        value = ["abc" * 200] * 6
        pretty_value = p._prettify_list(value)
        mock.assert_has_calls([call("abc" * 200)] * p.max_list_length)
        assert pretty_value == ["mocked string"] * 5 + ["... (1 items)"]


def test_prettifier_dict_nested() -> None:
    with patch.object(
        Prettifier, "_prettify_str", return_value="mocked string"
    ) as mock_str:
        with patch.object(
            Prettifier, "_prettify_list", return_value=["mocked list"]
        ) as mock_list:
            p = Prettifier()
            value = {
                "key1": "string",
                "key2": ["a", "list"],
            }
            pretty_value = p._prettify_dict(value)
            mock_str.assert_has_calls([call("string")])
            mock_list.assert_has_calls(
                [
                    call(["a", "list"]),
                ]
            )
            assert pretty_value == {
                "key1": "mocked string",
                "key2": ["mocked list"],
            }


def test_prettify_function() -> None:
    assert prettify(
        {
            "key": {
                "key0.1": "ab" * 200,
                "key0.2": ["a"] * 10,
                "key0.3": {"key0.3.1": "a short string"},
            }
        }
    ) == {
        "key": {
            "key0.1": "ab" * 100 + "... (200 chars)",
            "key0.2": ["a"] * 5 + ["... (5 items)"],
            "key0.3": {"key0.3.1": "a short string"},
        }
    }
