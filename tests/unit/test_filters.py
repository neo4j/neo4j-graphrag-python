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
from unittest.mock import patch, call

import pytest

from neo4j_genai.filters import (
    get_metadata_filter,
    _single_condition_cypher,
    _handle_field_filter,
    _construct_metadata_filter,
    EqOperator,
    NeqOperator,
    LtOperator,
    GtOperator,
    LteOperator,
    GteOperator,
    InOperator,
    NinOperator,
    LikeOperator,
    ILikeOperator,
    ParameterStore,
)


@pytest.fixture(scope="function")
def param_store_empty():
    return ParameterStore()


def test_param_store():
    ps = ParameterStore()
    assert ps.params == {}
    ps.add("", 1)
    assert ps.params == {"param_0": 1}
    ps.add("", "some value")
    assert ps.params == {"param_0": 1, "param_1": "some value"}


def test_single_condition_cypher_eq(param_store_empty):
    generated = _single_condition_cypher(
        "field", EqOperator, "value", param_store=param_store_empty
    )
    assert generated == "node.`field` = $param_0"
    assert param_store_empty.params == {"param_0": "value"}


def test_single_condition_cypher_eq_node_alias(param_store_empty):
    generated = _single_condition_cypher(
        "field", EqOperator, "value", node_alias="n", param_store=param_store_empty
    )
    assert generated == "n.`field` = $param_0"
    assert param_store_empty.params == {"param_0": "value"}


def test_single_condition_cypher_neq(param_store_empty):
    generated = _single_condition_cypher(
        "field", NeqOperator, "value", param_store=param_store_empty
    )
    assert generated == "node.`field` <> $param_0"
    assert param_store_empty.params == {"param_0": "value"}


def test_single_condition_cypher_lt(param_store_empty):
    generated = _single_condition_cypher(
        "field", LtOperator, 10, param_store=param_store_empty
    )
    assert generated == "node.`field` < $param_0"
    assert param_store_empty.params == {"param_0": 10}


def test_single_condition_cypher_gt(param_store_empty):
    generated = _single_condition_cypher(
        "field", GtOperator, 10, param_store=param_store_empty
    )
    assert generated == "node.`field` > $param_0"
    assert param_store_empty.params == {"param_0": 10}


def test_single_condition_cypher_lte(param_store_empty):
    generated = _single_condition_cypher(
        "field", LteOperator, 10, param_store=param_store_empty
    )
    assert generated == "node.`field` <= $param_0"
    assert param_store_empty.params == {"param_0": 10}


def test_single_condition_cypher_gte(param_store_empty):
    generated = _single_condition_cypher(
        "field", GteOperator, 10, param_store=param_store_empty
    )
    assert generated == "node.`field` >= $param_0"
    assert param_store_empty.params == {"param_0": 10}


def test_single_condition_cypher_in_int(param_store_empty):
    generated = _single_condition_cypher(
        "field", InOperator, [1, 2, 3], param_store=param_store_empty
    )
    assert generated == "node.`field` IN $param_0"
    assert param_store_empty.params == {"param_0": [1, 2, 3]}


def test_single_condition_cypher_in_str(param_store_empty):
    generated = _single_condition_cypher(
        "field", InOperator, ["a", "b", "c"], param_store=param_store_empty
    )
    assert generated == "node.`field` IN $param_0"
    assert param_store_empty.params == {"param_0": ["a", "b", "c"]}


def test_single_condition_cypher_in_invalid_type(param_store_empty):
    with pytest.raises(ValueError) as excinfo:
        _single_condition_cypher(
            "field",
            InOperator,
            [
                {"my_tuple"},
            ],
            param_store=param_store_empty,
        )
    assert "Unsupported type: <class 'set'>" in str(excinfo)


def test_single_condition_cypher_nin(param_store_empty):
    generated = _single_condition_cypher(
        "field", NinOperator, ["a", "b", "c"], param_store=param_store_empty
    )
    assert generated == "node.`field` NOT IN $param_0"
    assert param_store_empty.params == {"param_0": ["a", "b", "c"]}


def test_single_condition_cypher_like(param_store_empty):
    generated = _single_condition_cypher(
        "field", LikeOperator, "value", param_store=param_store_empty
    )
    assert generated == "node.`field` CONTAINS $param_0"
    assert param_store_empty.params == {"param_0": "value"}


def test_single_condition_cypher_ilike(param_store_empty):
    generated = _single_condition_cypher(
        "field", ILikeOperator, "My Value", param_store=param_store_empty
    )
    assert generated == "toLower(node.`field`) CONTAINS $param_0"
    assert param_store_empty.params == {"param_0": "my value"}


def test_single_condition_cypher_like_not_a_string(param_store_empty):
    with pytest.raises(ValueError) as excinfo:
        _single_condition_cypher(
            "field", ILikeOperator, 1, param_store=param_store_empty
        )
    assert "Expected string value, got <class 'int'>" in str(excinfo)


def test_handle_field_filter_not_a_string(param_store_empty):
    with pytest.raises(ValueError) as excinfo:
        _handle_field_filter(1, "value", param_store=param_store_empty)
    assert "Field should be a string but got: <class 'int'> with value: 1" in str(
        excinfo
    )


def test_handle_field_filter_field_start_with_dollar_sign(param_store_empty):
    with pytest.raises(ValueError) as excinfo:
        _handle_field_filter("$field_name", "value", param_store=param_store_empty)
    assert (
        "Invalid filter condition. Expected a field but got an operator: $field_name"
        in str(excinfo)
    )


def test_handle_field_filter_bad_field_name(param_store_empty):
    with pytest.raises(ValueError) as excinfo:
        _handle_field_filter("bad+field?name", "value", param_store=param_store_empty)
    assert "Invalid field name: bad+field?name. Expected a valid identifier." in str(
        excinfo
    )


def test_handle_field_filter_bad_value(param_store_empty):
    with pytest.raises(ValueError) as excinfo:
        _handle_field_filter(
            "field",
            value={"operator1": "value1", "operator2": "value2"},
            param_store=param_store_empty,
        )
    assert "Invalid filter condition" in str(excinfo)


def test_handle_field_filter_bad_operator_name(param_store_empty):
    with pytest.raises(ValueError) as excinfo:
        _handle_field_filter(
            "field", value={"$invalid": "value"}, param_store=param_store_empty
        )
    assert "Invalid operator: $invalid" in str(excinfo)


def test_handle_field_filter_operator_between(param_store_empty):
    generated = _handle_field_filter(
        "field", value={"$between": [0, 1]}, param_store=param_store_empty
    )
    assert generated == "$param_0 <= node.`field` <= $param_1"
    assert param_store_empty.params == {"param_0": 0, "param_1": 1}


def test_handle_field_filter_operator_between_not_enough_parameters(param_store_empty):
    with pytest.raises(ValueError) as excinfo:
        _handle_field_filter(
            "field",
            value={
                "$between": [
                    0,
                ]
            },
            param_store=param_store_empty,
        )
    assert "Expected lower and upper bounds in a list, got [0]" in str(excinfo)


@patch("neo4j_genai.filters._single_condition_cypher", return_value="condition")
def test_handle_field_filter_implicit_eq(
    _single_condition_cypher_mocked, param_store_empty
):
    generated = _handle_field_filter(
        "field", value="some_value", param_store=param_store_empty
    )
    _single_condition_cypher_mocked.assert_called_once_with(
        "field", EqOperator, "some_value", param_store_empty, "node"
    )
    assert generated == "condition"


@patch("neo4j_genai.filters._single_condition_cypher")
def test_handle_field_filter_eq(_single_condition_cypher_mocked, param_store_empty):
    _handle_field_filter(
        "field", value={"$eq": "some_value"}, param_store=param_store_empty
    )
    _single_condition_cypher_mocked.assert_called_once_with(
        "field", EqOperator, "some_value", param_store_empty, "node"
    )


@patch("neo4j_genai.filters._single_condition_cypher")
def test_handle_field_filter_neq(_single_condition_cypher_mocked, param_store_empty):
    _handle_field_filter(
        "field", value={"$ne": "some_value"}, param_store=param_store_empty
    )
    _single_condition_cypher_mocked.assert_called_once_with(
        "field", NeqOperator, "some_value", param_store_empty, "node"
    )


@patch("neo4j_genai.filters._single_condition_cypher")
def test_handle_field_filter_lt(_single_condition_cypher_mocked, param_store_empty):
    _handle_field_filter("field", value={"$lt": 1}, param_store=param_store_empty)
    _single_condition_cypher_mocked.assert_called_once_with(
        "field", LtOperator, 1, param_store_empty, "node"
    )


@patch("neo4j_genai.filters._single_condition_cypher")
def test_handle_field_filter_gt(_single_condition_cypher_mocked, param_store_empty):
    _handle_field_filter("field", value={"$gt": 1}, param_store=param_store_empty)
    _single_condition_cypher_mocked.assert_called_once_with(
        "field", GtOperator, 1, param_store_empty, "node"
    )


@patch("neo4j_genai.filters._single_condition_cypher")
def test_handle_field_filter_lte(_single_condition_cypher_mocked, param_store_empty):
    _handle_field_filter("field", value={"$lte": 1}, param_store=param_store_empty)
    _single_condition_cypher_mocked.assert_called_once_with(
        "field", LteOperator, 1, param_store_empty, "node"
    )


@patch("neo4j_genai.filters._single_condition_cypher")
def test_handle_field_filter_gte(_single_condition_cypher_mocked, param_store_empty):
    _handle_field_filter("field", value={"$gte": 1}, param_store=param_store_empty)
    _single_condition_cypher_mocked.assert_called_once_with(
        "field", GteOperator, 1, param_store_empty, "node"
    )


@patch("neo4j_genai.filters._single_condition_cypher")
def test_handle_field_filter_in(_single_condition_cypher_mocked, param_store_empty):
    _handle_field_filter("field", value={"$in": [1, 2]}, param_store=param_store_empty)
    _single_condition_cypher_mocked.assert_called_once_with(
        "field", InOperator, [1, 2], param_store_empty, "node"
    )


@patch("neo4j_genai.filters._single_condition_cypher")
def test_handle_field_filter_nin(_single_condition_cypher_mocked, param_store_empty):
    _handle_field_filter("field", value={"$nin": [1, 2]}, param_store=param_store_empty)
    _single_condition_cypher_mocked.assert_called_once_with(
        "field", NinOperator, [1, 2], param_store_empty, "node"
    )


@patch("neo4j_genai.filters._single_condition_cypher")
def test_handle_field_filter_like(_single_condition_cypher_mocked, param_store_empty):
    _handle_field_filter(
        "field", value={"$like": "value"}, param_store=param_store_empty
    )
    _single_condition_cypher_mocked.assert_called_once_with(
        "field", LikeOperator, "value", param_store_empty, "node"
    )


@patch("neo4j_genai.filters._single_condition_cypher")
def test_handle_field_filter_ilike(_single_condition_cypher_mocked, param_store_empty):
    _handle_field_filter(
        "field", value={"$ilike": "value"}, param_store=param_store_empty
    )
    _single_condition_cypher_mocked.assert_called_once_with(
        "field", ILikeOperator, "value", param_store_empty, "node"
    )


@patch("neo4j_genai.filters._handle_field_filter")
def test_construct_metadata_filter_filter_is_not_a_dict(_handle_field_filter_mock, param_store_empty):
    with pytest.raises(ValueError) as excinfo:
        _construct_metadata_filter([], param_store_empty, node_alias="n")
    assert "Filter must be a dictionary, got <class 'list'>" in str(excinfo)


@patch("neo4j_genai.filters._handle_field_filter")
def test_construct_metadata_filter_no_operator(_handle_field_filter_mock, param_store_empty):
    _construct_metadata_filter({"field": "value"}, param_store_empty, node_alias="n")
    _handle_field_filter_mock.assert_called_once_with(
        "field", "value", param_store_empty, node_alias="n"
    )


@patch("neo4j_genai.filters._construct_metadata_filter")
def test_construct_metadata_filter_implicit_and(_construct_metadata_filter_mock, param_store_empty):
    _construct_metadata_filter({"field_1": "value_1", "field_2": "value_2"}, param_store_empty, node_alias="n")
    _construct_metadata_filter_mock.assert_has_calls([
        call({"$and": [{"field_1": "value_1"}, {"field_2": "value_2"}]}, param_store_empty, "n"),
    ])


@patch("neo4j_genai.filters._construct_metadata_filter", side_effect=["filter1", "filter2"])
def test_construct_metadata_filter_explicit_and(_construct_metadata_filter_mock, param_store_empty):
    generated = _construct_metadata_filter({"$and": [{"field_1": "value_1"}, {"field_2": "value_2"}]}, param_store_empty, node_alias="n")
    _construct_metadata_filter_mock.assert_has_calls([
        call({"field_1": "value_1"}, param_store_empty, "n"),
        call({"field_2": "value_2"}, param_store_empty, "n")
    ])
    assert generated == "(filter1) AND (filter2)"


@patch("neo4j_genai.filters._construct_metadata_filter", side_effect=["filter1", "filter2"])
def test_construct_metadata_filter_or(_construct_metadata_filter_mock, param_store_empty):
    generated = _construct_metadata_filter({"$or": [{"field_1": "value_1"}, {"field_2": "value_2"}]}, param_store_empty, node_alias="n")
    _construct_metadata_filter_mock.assert_has_calls([
        call({"field_1": "value_1"}, param_store_empty, "n"),
        call({"field_2": "value_2"}, param_store_empty, "n")
    ])
    assert generated == "(filter1) OR (filter2)"


def test_construct_metadata_filter_invalid_operator(param_store_empty):
    with pytest.raises(ValueError) as excinfo:
        _construct_metadata_filter({"$invalid": [{}, {}]}, param_store_empty, node_alias="n")
    assert "Unsupported operator: $invalid" in str(excinfo)


def test_get_metadata_filter_single_field_string():
    filters = {"field": "string_value"}
    query, params = get_metadata_filter(filters)
    assert query == "node.`field` = $param_0"
    assert params == {"param_0": "string_value"}


def test_get_metadata_filter_single_field_int():
    filters = {"field": 28}
    query, params = get_metadata_filter(filters)
    assert query == "node.`field` = $param_0"
    assert params == {"param_0": 28}


def test_get_metadata_filter_single_field_bool():
    filters = {"field": False}
    query, params = get_metadata_filter(filters)
    assert query == "node.`field` = $param_0"
    assert params == {"param_0": False}


def test_get_metadata_filter_explicit_eq_operator():
    filters = {"field": {"$eq": "string_value"}}
    query, params = get_metadata_filter(filters)
    assert query == "node.`field` = $param_0"
    assert params == {"param_0": "string_value"}


def test_get_metadata_filter_neq_operator():
    filters = {"field": {"$ne": "string_value"}}
    query, params = get_metadata_filter(filters)
    assert query == "node.`field` <> $param_0"
    assert params == {"param_0": "string_value"}


def test_get_metadata_filter_lt_operator():
    filters = {"field": {"$lt": 1}}
    query, params = get_metadata_filter(filters)
    assert query == "node.`field` < $param_0"
    assert params == {"param_0": 1}


def test_get_metadata_filter_gt_operator():
    filters = {"field": {"$gt": 1}}
    query, params = get_metadata_filter(filters)
    assert query == "node.`field` > $param_0"
    assert params == {"param_0": 1}


def test_get_metadata_filter_lte_operator():
    filters = {"field": {"$lte": 1}}
    query, params = get_metadata_filter(filters)
    assert query == "node.`field` <= $param_0"
    assert params == {"param_0": 1}


def test_get_metadata_filter_gte_operator():
    filters = {"field": {"$gte": 1}}
    query, params = get_metadata_filter(filters)
    assert query == "node.`field` >= $param_0"
    assert params == {"param_0": 1}


def test_get_metadata_filter_in_operator():
    filters = {"field": {"$in": ["a", "b"]}}
    query, params = get_metadata_filter(filters)
    assert query == "node.`field` IN $param_0"
    assert params == {"param_0": ["a", "b"]}


def test_get_metadata_filter_not_in_operator():
    filters = {"field": {"$nin": ["a", "b"]}}
    query, params = get_metadata_filter(filters)
    assert query == "node.`field` NOT IN $param_0"
    assert params == {"param_0": ["a", "b"]}


def test_get_metadata_filter_like_operator():
    filters = {"field": {"$like": "some_value"}}
    query, params = get_metadata_filter(filters)
    assert query == "node.`field` CONTAINS $param_0"
    assert params == {"param_0": "some_value"}


def test_get_metadata_filter_ilike_operator():
    filters = {"field": {"$ilike": "Some Value"}}
    query, params = get_metadata_filter(filters)
    assert query == "toLower(node.`field`) CONTAINS $param_0"
    assert params == {"param_0": "some value"}


def test_get_metadata_filter_between_operator():
    filters = {"field": {"$between": [0, 1]}}
    query, params = get_metadata_filter(filters)
    assert query == "$param_0 <= node.`field` <= $param_1"
    assert params == {"param_0": 0, "param_1": 1}


def test_get_metadata_filter_implicit_and_condition():
    filters = {"field_1": "string_value", "field_2": True}
    query, params = get_metadata_filter(filters)
    assert query == "(node.`field_1` = $param_0) AND (node.`field_2` = $param_1)"
    assert params == {"param_0": "string_value", "param_1": True}


def test_get_metadata_filter_explicit_and_condition():
    filters = {"$and": [{"field_1": "string_value"}, {"field_2": True}]}
    query, params = get_metadata_filter(filters)
    assert query == "(node.`field_1` = $param_0) AND (node.`field_2` = $param_1)"
    assert params == {"param_0": "string_value", "param_1": True}


def test_get_metadata_filter_explicit_and_condition_with_operator():
    filters = {
        "$and": [{"field_1": {"$ne": "string_value"}}, {"field_2": {"$in": [1, 2]}}]
    }
    query, params = get_metadata_filter(filters)
    assert query == "(node.`field_1` <> $param_0) AND (node.`field_2` IN $param_1)"
    assert params == {"param_0": "string_value", "param_1": [1, 2]}


def test_get_metadata_filter_or_condition():
    filters = {"$or": [{"field_1": "string_value"}, {"field_2": True}]}
    query, params = get_metadata_filter(filters)
    assert query == "(node.`field_1` = $param_0) OR (node.`field_2` = $param_1)"
    assert params == {"param_0": "string_value", "param_1": True}


def test_get_metadata_filter_and_or_combined():
    filters = {
        "$and": [
            {"$or": [{"field_1": "string_value"}, {"field_2": True}]},
            {"field_3": 11},
        ]
    }
    query, params = get_metadata_filter(filters)
    assert query == (
        "((node.`field_1` = $param_0) OR (node.`field_2` = $param_1)) "
        "AND (node.`field_3` = $param_2)"
    )
    assert params == {"param_0": "string_value", "param_1": True, "param_2": 11}


# now testing bad filters
def test_get_metadata_filter_field_name_with_dollar_sign():
    filters = {"$field": "value"}
    with pytest.raises(ValueError):
        get_metadata_filter(filters)


def test_get_metadata_filter_and_no_list():
    filters = {"$and": {}}
    with pytest.raises(ValueError):
        get_metadata_filter(filters)


def test_get_metadata_filter_unsupported_operator():
    filters = {"field": {"$unsupported": "value"}}
    with pytest.raises(ValueError):
        get_metadata_filter(filters)
