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
"""
Filters format:
{"property_name": "property_value"}


"""
from typing import Any, Type
from collections import Counter


DEFAULT_NODE_ALIAS = "node"


class Operator:
    """Operator classes are helper classes to build the Cypher queries
    from a filter like {"field_name": "field_value"}
    They implement two important methods:
    - lhs: (left hand side): the node + property to be filtered on
        + optional operations on it (see ILikeOperator for instance)
    - cleaned_value: a method to make sure the provided parameter values are
        consistent with the operator (e.g. LIKE operator only works with string values)
    """
    CYPHER_OPERATOR = None

    def __init__(self, node_alias=DEFAULT_NODE_ALIAS):
        self.node_alias = node_alias

    def lhs(self, field):
        return f"{self.node_alias}.`{field}`"

    def cleaned_value(self, value):
        return value


class EqOperator(Operator):
    CYPHER_OPERATOR = "="


class NeqOperator(Operator):
    CYPHER_OPERATOR = "<>"


class LtOperator(Operator):
    CYPHER_OPERATOR = "<"


class GtOperator(Operator):
    CYPHER_OPERATOR = ">"


class LteOperator(Operator):
    CYPHER_OPERATOR = "<="


class GteOperator(Operator):
    CYPHER_OPERATOR = ">="


class InOperator(Operator):
    CYPHER_OPERATOR = "IN"

    def cleaned_value(self, value):
        for val in value:
            if not isinstance(val, (str, int, float)):
                raise NotImplementedError(
                    f"Unsupported type: {type(val)} for value: {val}"
                )
        return value


class NinOperator(InOperator):
    CYPHER_OPERATOR = "NOT IN"


class LikeOperator(Operator):
    CYPHER_OPERATOR = "CONTAINS"

    def cleaned_value(self, value):
        if not isinstance(value, str):
            raise ValueError(f"Expected string value, got {type(value)}: {value}")
        return value.rstrip("%")


class ILikeOperator(LikeOperator):

    def lhs(self, field):
        return f"toLower({self.node_alias}.`{field}`)"

    def cleaned_value(self, value):
        value = super().cleaned_value(value)
        return value.lower()


OPERATOR_PREFIX = "$"

OPERATOR_EQ = "$eq"
OPERATOR_NE = "$ne"
OPERATOR_LT = "$lt"
OPERATOR_LTE = "$lte"
OPERATOR_GT = "$gt"
OPERATOR_GTE = "$gte"
OPERATOR_BETWEEN = "$between"
OPERATOR_IN = "$in"
OPERATOR_NIN = "$nin"
OPERATOR_LIKE = "$like"
OPERATOR_ILIKE = "$ilike"

OPERATOR_AND = "$and"
OPERATOR_OR = "$or"

COMPARISONS_TO_NATIVE = {
    OPERATOR_EQ: EqOperator,
    OPERATOR_NE: NeqOperator,
    OPERATOR_LT: LtOperator,
    OPERATOR_LTE: LteOperator,
    OPERATOR_GT: GtOperator,
    OPERATOR_GTE: GteOperator,
    OPERATOR_IN: InOperator,
    OPERATOR_NIN: NinOperator,
    OPERATOR_LIKE: LikeOperator,
    OPERATOR_ILIKE: ILikeOperator,
}


LOGICAL_OPERATORS = {OPERATOR_AND, OPERATOR_OR}

SUPPORTED_OPERATORS = (
    set(COMPARISONS_TO_NATIVE)
    .union(LOGICAL_OPERATORS)
    .union({OPERATOR_BETWEEN})
)


class ParameterStore:
    """
    Store parameters for a given query.
    Determine the parameter name depending on a parameter counter
    """

    def __init__(self):
        self._counter = Counter()
        self.params = {}

    def _get_params_name(self, key="param"):
        """NB: the counter parameter is there in purpose, will be modified in the function
        to remember the count of each parameter

        :param p:
        :param counter:
        :return:
        """
        # key = slugify(key.replace(".", "_"), separator="_")
        param_name = f"{key}_{self._counter[key]}"
        self._counter[key] += 1
        return param_name

    def add(self, key, value):
        param_name = self._get_params_name()
        self.params[param_name] = value
        return param_name


def _single_condition_cypher(field: str, native_operator_class: Type[Operator], value: Any, param_store: ParameterStore, node_alias: str) -> str:
    """Return Cypher for field operator value
    NB: the param_store argument is mutable, it will be updated in this function
    """
    native_op = native_operator_class()
    param_name = param_store.add(field, native_op.cleaned_value(value))
    query_snippet = f"{native_op.lhs(field)} {native_op.CYPHER_OPERATOR} ${param_name}"
    return query_snippet


def _handle_field_filter(
    field: str, value: Any, param_store: ParameterStore,
    node_alias: str = DEFAULT_NODE_ALIAS
) -> str:
    """Create a filter for a specific field.

    Args:
        field: name of field
        value: value to filter
            If provided as is then this will be an equality filter
            If provided as a dictionary then this will be a filter, the key
            will be the operator and the value will be the value to filter by
        param_store:
        node_alias:

    Returns
        - Cypher filter snippet*

    NB: the param_store argument is mutable, it will be updated in this function
    """
    # first, perform some sanity checks
    if not isinstance(field, str):
        raise ValueError(
            f"Field should be a string but got: {type(field)} with value: {field}"
        )

    if field.startswith(OPERATOR_PREFIX):
        raise ValueError(
            f"Invalid filter condition. Expected a field but got an operator: "
            f"{field}"
        )

    # Allow [a-zA-Z0-9_], disallow $ for now until we support escape characters
    if not field.isidentifier():
        raise ValueError(f"Invalid field name: {field}. Expected a valid identifier.")

    if isinstance(value, dict):
        # This is a filter specification e.g. {"$gte": 0}
        if len(value) != 1:
            raise ValueError(
                "Invalid filter condition. Expected a value which "
                "is a dictionary with a single key that corresponds to an operator "
                f"but got a dictionary with {len(value)} keys. The first few "
                f"keys are: {list(value.keys())[:3]}"
            )
        operator, filter_value = list(value.items())[0]
        operator = operator.lower()
        # Verify that that operator is an operator
        if operator not in SUPPORTED_OPERATORS:
            raise ValueError(
                f"Invalid operator: {operator}. "
                f"Expected one of {SUPPORTED_OPERATORS}"
            )
    else:  # if value is not dict, then we assume an equality operator
        operator = OPERATOR_EQ
        filter_value = value

    # now everything is set, we can start and build the query
    # special case for the BETWEEN operator that requires
    # two tests (lower_bound <= value <= higher_bound)
    if operator == OPERATOR_BETWEEN:
        low, high = filter_value
        param_name_low = param_store.add(field, low)
        param_name_high = param_store.add(field, high)
        query_snippet = (
            f"${param_name_low} <= {DEFAULT_NODE_ALIAS}.`{field}` <= ${param_name_high}"
        )
        return query_snippet
    # all the other operators are handled through their own classes:
    native_op_class = COMPARISONS_TO_NATIVE[operator]
    return _single_condition_cypher(field, native_op_class, filter_value, param_store, node_alias)


def _construct_metadata_filter(filter: dict[str, Any], param_store: ParameterStore, node_alias: str) -> str:
    """Construct a metadata filter. This is a recursive function parsing the filter dict

    Args:
        filter: A dictionary representing the filter condition.
        param_store: A ParamStore object that will deal with parameter naming and saving along the process
        node_alias: a string used as alias for the node the filters will be applied to (must come from earlier in the query)

    Returns:
        str

    NB: the param_store argument is mutable, it will be updated in this function
    """

    if not isinstance(filter, dict):
        raise ValueError()
    # if we have more than one entry, this is an implicit "AND" filter
    if len(filter) > 1:
        return _construct_metadata_filter({OPERATOR_AND: [{k: v} for k, v in filter.items()]}, param_store, node_alias)
    # The only operators allowed at the top level are $AND and $OR
    # First check if an operator or a field
    key, value = list(filter.items())[0]
    if not key.startswith("$"):
        # it's not an operator, must be a field
        return _handle_field_filter(key, filter[key], param_store, node_alias=node_alias)

    # Here we handle the $and and $or operators
    if not isinstance(value, list):
        raise ValueError(
            f"Expected a list, but got {type(value)} for value: {value}"
        )
    if key.lower() == OPERATOR_AND:
        cypher_operator = " AND "
    elif key.lower() == OPERATOR_OR:
        cypher_operator = " OR "
    else:
        raise ValueError(f"Unsupported filter {filter}")
    query = cypher_operator.join(
        [f"({ _construct_metadata_filter(el, param_store, node_alias)})" for el in value]
    )
    return query


def construct_metadata_filter(filter: dict[str, Any], node_alias: str = DEFAULT_NODE_ALIAS) -> tuple[str, dict]:
    """Construct the cypher filter snippet based on a filter dict

    Args:
        filter: a dict of filters
        node_alias: the node the filters must be applied on

    Return:
        A tuple of str, dict where the string is the cypher query and the dict
        contains the query parameters
    """
    param_store = ParameterStore()
    return _construct_metadata_filter(filter, param_store, node_alias=node_alias), param_store.params
