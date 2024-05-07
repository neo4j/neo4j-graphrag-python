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
import pytest

from neo4j_genai.filters import get_metadata_filter


def test_filter_single_field_string():
    filters = {"field": "string_value"}
    query, params = get_metadata_filter(filters)
    assert query == "node.`field` = $param_0"
    assert params == {"param_0": "string_value"}


def test_filter_single_field_int():
    filters = {"field": 28}
    query, params = get_metadata_filter(filters)
    assert query == "node.`field` = $param_0"
    assert params == {"param_0": 28}


def test_filter_single_field_bool():
    filters = {"field": False}
    query, params = get_metadata_filter(filters)
    assert query == "node.`field` = $param_0"
    assert params == {"param_0": False}


def test_filter_explicit_eq_operator():
    filters = {"field": {"$eq": "string_value"}}
    query, params = get_metadata_filter(filters)
    assert query == "node.`field` = $param_0"
    assert params == {"param_0": "string_value"}


def test_filter_neq_operator():
    filters = {"field": {"$ne": "string_value"}}
    query, params = get_metadata_filter(filters)
    assert query == "node.`field` <> $param_0"
    assert params == {"param_0": "string_value"}


def test_filter_lt_operator():
    filters = {"field": {"$lt": 1}}
    query, params = get_metadata_filter(filters)
    assert query == "node.`field` < $param_0"
    assert params == {"param_0": 1}


def test_filter_gt_operator():
    filters = {"field": {"$gt": 1}}
    query, params = get_metadata_filter(filters)
    assert query == "node.`field` > $param_0"
    assert params == {"param_0": 1}


def test_filter_lte_operator():
    filters = {"field": {"$lte": 1}}
    query, params = get_metadata_filter(filters)
    assert query == "node.`field` <= $param_0"
    assert params == {"param_0": 1}


def test_filter_gte_operator():
    filters = {"field": {"$gte": 1}}
    query, params = get_metadata_filter(filters)
    assert query == "node.`field` >= $param_0"
    assert params == {"param_0": 1}


def test_filter_in_operator():
    filters = {"field": {"$in": ["a", "b"]}}
    query, params = get_metadata_filter(filters)
    assert query == "node.`field` IN $param_0"
    assert params == {"param_0": ["a", "b"]}


def test_filter_not_in_operator():
    filters = {"field": {"$nin": ["a", "b"]}}
    query, params = get_metadata_filter(filters)
    assert query == "node.`field` NOT IN $param_0"
    assert params == {"param_0": ["a", "b"]}


def test_filter_like_operator():
    filters = {"field": {"$like": "some_value"}}
    query, params = get_metadata_filter(filters)
    assert query == "node.`field` CONTAINS $param_0"
    assert params == {"param_0": "some_value"}


def test_filter_ilike_operator():
    filters = {"field": {"$ilike": "Some Value"}}
    query, params = get_metadata_filter(filters)
    assert query == "toLower(node.`field`) CONTAINS $param_0"
    assert params == {"param_0": "some value"}


def test_filter_between_operator():
    filters = {"field": {"$between": [0, 1]}}
    query, params = get_metadata_filter(filters)
    assert query == "$param_0 <= node.`field` <= $param_1"
    assert params == {"param_0": 0, "param_1": 1}


def test_filter_implicit_and_condition():
    filters = {"field_1": "string_value", "field_2": True}
    query, params = get_metadata_filter(filters)
    assert query == "(node.`field_1` = $param_0) AND (node.`field_2` = $param_1)"
    assert params == {"param_0": "string_value", "param_1": True}


def test_filter_explicit_and_condition():
    filters = {"$and": [{"field_1": "string_value"}, {"field_2": True}]}
    query, params = get_metadata_filter(filters)
    assert query == "(node.`field_1` = $param_0) AND (node.`field_2` = $param_1)"
    assert params == {"param_0": "string_value", "param_1": True}


def test_filter_or_condition():
    filters = {"$or": [{"field_1": "string_value"}, {"field_2": True}]}
    query, params = get_metadata_filter(filters)
    assert query == "(node.`field_1` = $param_0) OR (node.`field_2` = $param_1)"
    assert params == {"param_0": "string_value", "param_1": True}


def test_filter_and_or_combined():
    filters = {
        "$and": [
            {"$or": [{"field_1": "string_value"}, {"field_2": True}]},
            {"field_3": 11},
        ]
    }
    query, params = get_metadata_filter(filters)
    assert (
        query
        == "((node.`field_1` = $param_0) OR (node.`field_2` = $param_1)) AND (node.`field_3` = $param_2)"
    )
    assert params == {"param_0": "string_value", "param_1": True, "param_2": 11}


# now testing bad filters
def test_field_name_with_dollar_sign():
    filters = {"$field": "value"}
    with pytest.raises(ValueError):
        get_metadata_filter(filters)


def test_and_no_list():
    filters = {"$and": {}}
    with pytest.raises(ValueError):
        get_metadata_filter(filters)


def test_unsupported_operator():
    filters = {"field": {"$unsupported": "value"}}
    with pytest.raises(ValueError):
        get_metadata_filter(filters)
