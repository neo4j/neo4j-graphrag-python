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
    # Note: 'required' is handled at the object level, not individual parameter level
    assert "required" not in d


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
    # Test that individual parameters don't include 'required' field (it's handled at object level)
    string_param = StringParameter(description="Required string", required=True)
    assert "required" not in string_param.model_dump_tool()

    integer_param = IntegerParameter(description="Required integer", required=True)
    assert "required" not in integer_param.model_dump_tool()

    number_param = NumberParameter(description="Required number", required=True)
    assert "required" not in number_param.model_dump_tool()

    boolean_param = BooleanParameter(description="Required boolean", required=True)
    assert "required" not in boolean_param.model_dump_tool()

    array_param = ArrayParameter(
        description="Required array",
        items=StringParameter(description="item"),
        required=True,
    )
    assert "required" not in array_param.model_dump_tool()

    object_param = ObjectParameter(
        description="Required object",
        properties={"prop": StringParameter(description="property")},
        required=True,
    )
    assert "required" not in object_param.model_dump_tool()

    # Test that optional parameters also don't include the required field
    optional_param = StringParameter(description="Optional string", required=False)
    assert "required" not in optional_param.model_dump_tool()


def test_object_parameter_additional_properties_always_present() -> None:
    """Test that additionalProperties is always present in ObjectParameter schema, fixing OpenAI compatibility."""

    # Test additionalProperties=True (default)
    obj_param_true = ObjectParameter(
        description="Object with additional properties",
        properties={"prop": StringParameter(description="A property")},
        additional_properties=True,
    )
    schema_true = obj_param_true.model_dump_tool()
    assert "additionalProperties" in schema_true
    assert schema_true["additionalProperties"] is True

    # Test additionalProperties=False
    obj_param_false = ObjectParameter(
        description="Object without additional properties",
        properties={"prop": StringParameter(description="A property")},
        additional_properties=False,
    )
    schema_false = obj_param_false.model_dump_tool()
    assert "additionalProperties" in schema_false
    assert schema_false["additionalProperties"] is False


def test_json_schema_compatibility() -> None:
    """Test that the generated schema is compatible with JSON Schema specification."""

    # Create a complex object with nested properties and required fields
    nested_obj = ObjectParameter(
        description="Nested object",
        properties={
            "nested_prop": StringParameter(description="Nested string"),
        },
        additional_properties=True,
    )

    main_obj = ObjectParameter(
        description="Main object",
        properties={
            "required_string": StringParameter(description="Required string"),
            "optional_number": NumberParameter(description="Optional number"),
            "nested_object": nested_obj,
        },
        required_properties=["required_string"],
        additional_properties=False,
    )

    schema = main_obj.model_dump_tool()

    # Verify JSON Schema structure
    assert schema["type"] == "object"
    assert "properties" in schema
    assert "required" in schema
    assert "additionalProperties" in schema

    # Check required is an array (not boolean on individual properties)
    assert isinstance(schema["required"], list)
    assert "required_string" in schema["required"]
    assert len(schema["required"]) == 1

    # Check individual properties don't have 'required' field
    for prop_name, prop_schema in schema["properties"].items():
        assert "required" not in prop_schema

    # Check additionalProperties is properly set at all levels
    assert schema["additionalProperties"] is False
    assert schema["properties"]["nested_object"]["additionalProperties"] is True


def test_text2cypher_retriever_schema_compatibility() -> None:
    """Test the specific schema structure that caused the OpenAI API error."""

    # Simulate the Text2CypherRetriever parameter structure
    prompt_params = ObjectParameter(
        description="Parameter prompt_params",
        properties={},
        additional_properties=True,  # This was missing in the original bug
    )

    t2c_params = ObjectParameter(
        description="Parameters for Text2CypherRetriever",
        properties={
            "query_text": StringParameter(description="Parameter query_text"),
            "prompt_params": prompt_params,
        },
        required_properties=["query_text"],
        additional_properties=False,
    )

    schema = t2c_params.model_dump_tool()

    # Verify the fix: prompt_params should have additionalProperties
    prompt_params_schema = schema["properties"]["prompt_params"]
    assert "additionalProperties" in prompt_params_schema
    assert prompt_params_schema["additionalProperties"] is True

    # Verify query_text doesn't have individual 'required' field
    query_text_schema = schema["properties"]["query_text"]
    assert "required" not in query_text_schema

    # Verify required array at object level
    assert schema["required"] == ["query_text"]


def test_exclude_parameter_in_object_schema() -> None:
    """Test that exclude parameter works correctly in ObjectParameter.model_dump_tool()."""

    obj_param = ObjectParameter(
        description="Test object",
        properties={
            "prop1": StringParameter(description="Property 1"),
            "prop2": IntegerParameter(description="Property 2"),
        },
        required_properties=["prop1"],
        additional_properties=True,
    )

    # Test excluding required field
    schema_no_required = obj_param.model_dump_tool(exclude=["required"])
    assert "required" not in schema_no_required
    assert "additionalProperties" in schema_no_required  # Should still be present

    # Test excluding additionalProperties field
    schema_no_additional = obj_param.model_dump_tool(exclude=["additional_properties"])
    assert "additionalProperties" not in schema_no_additional
    assert "required" in schema_no_additional  # Should still be present


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
