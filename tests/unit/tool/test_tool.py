import pytest
from typing import Any
from neo4j_graphrag.tool import (
    StringParameter,
    IntegerParameter,
    NumberParameter,
    BooleanParameter,
    ArrayParameter,
    ObjectParameter,
    Tool,
    ToolParameter,
    ParameterType,
)


def test_string_parameter() -> None:
    param = StringParameter(description="A string", required=True, enum=["a", "b"])
    assert param.description == "A string"
    assert param.required is True
    assert param.enum == ["a", "b"]
    d = param.model_dump_tool()
    assert d["type"] == ParameterType.STRING
    assert d["enum"] == ["a", "b"]
    assert d["required"] is True


def test_integer_parameter() -> None:
    param = IntegerParameter(description="An int", minimum=0, maximum=10)
    d = param.model_dump_tool()
    assert d["type"] == ParameterType.INTEGER
    assert d["minimum"] == 0
    assert d["maximum"] == 10


def test_number_parameter() -> None:
    param = NumberParameter(description="A number", minimum=1.5, maximum=3.5)
    d = param.model_dump_tool()
    assert d["type"] == ParameterType.NUMBER
    assert d["minimum"] == 1.5
    assert d["maximum"] == 3.5


def test_boolean_parameter() -> None:
    param = BooleanParameter(description="A bool")
    d = param.model_dump_tool()
    assert d["type"] == ParameterType.BOOLEAN
    assert d["description"] == "A bool"


def test_array_parameter_and_validation() -> None:
    arr_param = ArrayParameter(
        description="An array",
        items=StringParameter(description="str"),
        min_items=1,
        max_items=5,
    )
    d = arr_param.model_dump_tool()
    assert d["type"] == ParameterType.ARRAY
    assert d["items"]["type"] == ParameterType.STRING
    assert d["minItems"] == 1
    assert d["maxItems"] == 5

    # Test items as dict
    arr_param2 = ArrayParameter(
        description="Arr with dict",
        items={"type": "string", "description": "str"},  # type: ignore
    )
    assert isinstance(arr_param2.items, StringParameter)

    # Test error on invalid items
    with pytest.raises(ValueError):
        # Use type: ignore to bypass type checking for this intentional error case
        ArrayParameter(description="bad", items=123).validate_items()  # type: ignore


def test_object_parameter_and_validation() -> None:
    obj_param = ObjectParameter(
        description="Obj",
        properties={
            "foo": StringParameter(description="foo"),
            "bar": IntegerParameter(description="bar"),
        },
        required_properties=["foo"],
        additional_properties=False,
    )
    d = obj_param.model_dump_tool()
    assert d["type"] == ParameterType.OBJECT
    assert d["properties"]["foo"]["type"] == ParameterType.STRING
    assert d["required"] == ["foo"]
    assert d["additionalProperties"] is False

    # Test properties as dicts
    obj_param2 = ObjectParameter(
        description="Obj2",
        properties={
            "foo": {"type": "string", "description": "foo"},  # type: ignore
        },
    )
    assert isinstance(obj_param2.properties["foo"], StringParameter)

    # Test error on invalid property
    with pytest.raises(ValueError):
        # Use type: ignore to bypass type checking for this intentional error case
        ObjectParameter(
            description="bad",
            properties={"foo": 123},  # type: ignore
        ).validate_properties()


def test_from_dict() -> None:
    d = {"type": ParameterType.STRING, "description": "desc"}
    param = ToolParameter.from_dict(d)
    assert isinstance(param, StringParameter)
    assert param.description == "desc"

    obj_dict = {
        "type": "object",
        "description": "obj",
        "properties": {"foo": {"type": "string", "description": "foo"}},
    }
    obj_param = ToolParameter.from_dict(obj_dict)
    assert isinstance(obj_param, ObjectParameter)
    assert isinstance(obj_param.properties["foo"], StringParameter)

    arr_dict = {
        "type": "array",
        "description": "arr",
        "items": {"type": "integer", "description": "int"},
    }
    arr_param = ToolParameter.from_dict(arr_dict)
    assert isinstance(arr_param, ArrayParameter)
    assert isinstance(arr_param.items, IntegerParameter)

    # Test unknown type
    with pytest.raises(ValueError):
        ToolParameter.from_dict({"type": "unknown", "description": "bad"})

    # Test missing type
    with pytest.raises(ValueError):
        ToolParameter.from_dict({"description": "no type"})


def test_required_parameter() -> None:
    # Test that required=True is included in model_dump_tool output for different parameter types
    string_param = StringParameter(description="Required string", required=True)
    assert string_param.model_dump_tool()["required"] is True

    integer_param = IntegerParameter(description="Required integer", required=True)
    assert integer_param.model_dump_tool()["required"] is True

    number_param = NumberParameter(description="Required number", required=True)
    assert number_param.model_dump_tool()["required"] is True

    boolean_param = BooleanParameter(description="Required boolean", required=True)
    assert boolean_param.model_dump_tool()["required"] is True

    array_param = ArrayParameter(
        description="Required array",
        items=StringParameter(description="item"),
        required=True,
    )
    assert array_param.model_dump_tool()["required"] is True

    object_param = ObjectParameter(
        description="Required object",
        properties={"prop": StringParameter(description="property")},
        required=True,
    )
    assert object_param.model_dump_tool()["required"] is True

    # Test that required=False doesn't include the required field
    optional_param = StringParameter(description="Optional string", required=False)
    assert "required" not in optional_param.model_dump_tool()


def test_tool_class() -> None:
    def dummy_func(**kwargs: Any) -> dict[str, Any]:
        return kwargs

    params = ObjectParameter(
        description="params",
        properties={"a": StringParameter(description="a")},
    )
    tool = Tool(
        name="mytool",
        description="desc",
        parameters=params,
        execute_func=dummy_func,
    )
    assert tool.get_name() == "mytool"
    assert tool.get_description() == "desc"
    assert tool.get_parameters()["type"] == ParameterType.OBJECT
    assert tool.execute(query="query", a="b") == {"query": "query", "a": "b"}

    # Test parameters as dict
    params_dict = {
        "type": "object",
        "description": "params",
        "properties": {"a": {"type": "string", "description": "a"}},
    }
    tool2 = Tool(
        name="mytool2",
        description="desc2",
        parameters=params_dict,
        execute_func=dummy_func,
    )
    assert tool2.get_parameters()["type"] == ParameterType.OBJECT
    assert tool2.execute(a="b") == {"a": "b"}
