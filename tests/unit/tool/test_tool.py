import pytest
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


def test_string_parameter():
    param = StringParameter(description="A string", required=True, enum=["a", "b"])
    assert param.description == "A string"
    assert param.required is True
    assert param.enum == ["a", "b"]
    d = param.model_dump_tool()
    assert d["type"] == ParameterType.STRING
    assert d["enum"] == ["a", "b"]


def test_integer_parameter():
    param = IntegerParameter(description="An int", minimum=0, maximum=10)
    d = param.model_dump_tool()
    assert d["type"] == ParameterType.INTEGER
    assert d["minimum"] == 0
    assert d["maximum"] == 10


def test_number_parameter():
    param = NumberParameter(description="A number", minimum=1.5, maximum=3.5)
    d = param.model_dump_tool()
    assert d["type"] == ParameterType.NUMBER
    assert d["minimum"] == 1.5
    assert d["maximum"] == 3.5


def test_boolean_parameter():
    param = BooleanParameter(description="A bool")
    d = param.model_dump_tool()
    assert d["type"] == ParameterType.BOOLEAN
    assert d["description"] == "A bool"


def test_array_parameter_and_validation():
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
        items={"type": "string", "description": "str"},
    )
    arr_param2 = arr_param2.validate_items()
    assert isinstance(arr_param2.items, StringParameter)

    # Test error on invalid items
    with pytest.raises(ValueError):
        ArrayParameter(description="bad", items=123).validate_items()


def test_object_parameter_and_validation():
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
            "foo": {"type": "string", "description": "foo"},
        },
    )
    obj_param2 = obj_param2.validate_properties()
    assert isinstance(obj_param2.properties["foo"], StringParameter)

    # Test error on invalid property
    with pytest.raises(ValueError):
        ObjectParameter(
            description="bad", properties={"foo": 123}
        ).validate_properties()


def test_from_dict():
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


def test_tool_class():
    def dummy_func(query, **kwargs):
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
    assert tool.execute("query", a="b") == {"a": "b"}

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
    assert tool2.execute("query", a="b") == {"a": "b"}
