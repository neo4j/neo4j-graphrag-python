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
from __future__ import annotations

import json
from typing import Tuple, Any
from unittest.mock import AsyncMock, patch, Mock

import pytest
from pydantic import ValidationError

from neo4j_graphrag.exceptions import SchemaValidationError, SchemaExtractionError
from neo4j_graphrag.experimental.components.schema import (
    SchemaBuilder,
    NodeType,
    PropertyType,
    RelationshipType,
    ConstraintType,
    SchemaFromTextExtractor,
    GraphSchema,
    SchemaFromExistingGraphExtractor,
)
import os
import tempfile
import yaml

from neo4j_graphrag.generation import PromptTemplate
from neo4j_graphrag.llm.types import LLMResponse
from neo4j_graphrag.utils.file_handler import FileFormat


def test_node_type_initialization_from_string() -> None:
    node_type = NodeType.model_validate("Label")
    assert isinstance(node_type, NodeType)
    assert node_type.label == "Label"
    assert node_type.properties == []


def test_node_type_additional_properties_default() -> None:
    # default behavior:
    node_type = NodeType.model_validate({"label": "Label"})
    assert node_type.additional_properties is True
    node_type = NodeType.model_validate({"label": "Label", "properties": []})
    assert node_type.additional_properties is True
    node_type = NodeType.model_validate(
        {"label": "Label", "properties": [{"name": "name", "type": "STRING"}]}
    )
    assert node_type.additional_properties is False

    # manually changing the default value
    # impossible cases: no properties and no additional
    with pytest.raises(ValidationError):
        NodeType.model_validate({"label": "Label", "additional_properties": False})
    with pytest.raises(ValidationError):
        NodeType.model_validate(
            {"label": "Label", "properties": [], "additional_properties": False}
        )

    # working case: properties and additional allowed
    node_type = NodeType.model_validate(
        {
            "label": "Label",
            "properties": [{"name": "name", "type": "STRING"}],
            "additional_properties": True,
        }
    )
    assert node_type.additional_properties is True


def test_relationship_type_initialization_from_string() -> None:
    relationship_type = RelationshipType.model_validate("REL")
    assert isinstance(relationship_type, RelationshipType)
    assert relationship_type.label == "REL"
    assert relationship_type.properties == []


def test_relationship_type_additional_properties_default() -> None:
    relationship_type = RelationshipType.model_validate({"label": "REL"})
    assert relationship_type.additional_properties is True
    relationship_type = RelationshipType.model_validate(
        {"label": "REL", "properties": []}
    )
    assert relationship_type.additional_properties is True
    relationship_type = RelationshipType.model_validate(
        {"label": "REL", "properties": [{"name": "name", "type": "STRING"}]}
    )
    assert relationship_type.additional_properties is False

    # manually changing the default value
    # impossible cases: no properties and no additional
    with pytest.raises(ValidationError):
        RelationshipType.model_validate(
            {"label": "REL", "additional_properties": False}
        )
    with pytest.raises(ValidationError):
        RelationshipType.model_validate(
            {"label": "REL", "properties": [], "additional_properties": False}
        )

    # working case: properties and additional allowed
    relationship_type = RelationshipType.model_validate(
        {
            "label": "REL",
            "properties": [{"name": "name", "type": "STRING"}],
            "additional_properties": True,
        }
    )
    assert relationship_type.additional_properties is True


def test_constraint_type_initialization() -> None:
    constraint = ConstraintType(
        type="UNIQUENESS", node_type="Person", property_name="name"
    )
    assert constraint.type == "UNIQUENESS"
    assert constraint.node_type == "Person"
    assert constraint.property_name == "name"


def test_constraint_type_is_frozen() -> None:
    constraint = ConstraintType(
        type="UNIQUENESS", node_type="Person", property_name="name"
    )

    with pytest.raises(ValidationError):
        constraint.type = "UNIQUENESS"

    with pytest.raises(ValidationError):
        constraint.node_type = "Organization"

    with pytest.raises(ValidationError):
        constraint.property_name = "id"


def test_schema_additional_node_types_default() -> None:
    schema_dict: dict[str, Any] = {
        "node_types": [],
    }
    schema = GraphSchema.model_validate(schema_dict)
    assert schema.additional_node_types is True

    schema_dict = {
        "node_types": ["Person"],
    }
    schema = GraphSchema.model_validate(schema_dict)
    assert schema.additional_node_types is False


def test_schema_additional_relationship_types_default() -> None:
    schema_dict: dict[str, Any] = {
        "node_types": [],
    }
    schema = GraphSchema.model_validate(schema_dict)
    assert schema.additional_relationship_types is True

    schema_dict = {
        "node_types": [],
        "relationship_types": ["REL"],
    }
    schema = GraphSchema.model_validate(schema_dict)
    assert schema.additional_relationship_types is False


def test_schema_additional_patterns_default() -> None:
    schema_dict: dict[str, Any] = {
        "node_types": [],
    }
    schema = GraphSchema.model_validate(schema_dict)
    assert schema.additional_patterns is True

    schema_dict = {
        "node_types": ["Person"],
        "relationship_types": ["REL"],
        "patterns": [("Person", "REL", "Person")],
    }
    schema = GraphSchema.model_validate(schema_dict)
    assert schema.additional_patterns is False


def test_schema_additional_parameter_validation() -> None:
    """Additional relationship types not allowed, but additional patterns allowed

    => raise Exception
    """
    schema_dict = {
        "node_types": [
            {
                "label": "Person",
                "properties": [
                    {
                        "name": "name",
                        "type": "STRING",
                    },
                    {"name": "height", "type": "INTEGER"},
                ],
            }
        ],
        "relationship_types": [
            {
                "label": "KNOWS",
            }
        ],
        "patterns": [
            ("Person", "KNOWS", "Person"),
        ],
        "additional_relationship_types": True,
        "additional_patterns": False,
    }
    with pytest.raises(
        ValidationError,
        match="`additional_relationship_types` must be set to False when using `additional_patterns=False`",
    ):
        GraphSchema.model_validate(schema_dict)


def test_schema_constraint_validation_property_not_in_node_type() -> None:
    schema_dict: dict[str, Any] = {
        "node_types": [
            {
                "label": "Person",
                "properties": [{"name": "name", "type": "STRING"}],
            }
        ],
        "constraints": [
            {"type": "UNIQUENESS", "node_type": "Person", "property_name": "email"}
        ],
    }

    with pytest.raises(SchemaValidationError) as exc_info:
        GraphSchema.model_validate(schema_dict)

    assert "Constraint references undefined property" in str(exc_info.value)
    assert "on node type 'Person'" in str(exc_info.value)


def test_schema_constraint_with_additional_properties_with_allows_unknown_property() -> (
    None
):
    # if additional_properties is True, we can define constraints that are not in the node_type
    schema_dict: dict[str, Any] = {
        "node_types": [
            {
                "label": "Person",
                "properties": [{"name": "name", "type": "STRING"}],
                "additional_properties": True,
            }
        ],
        "constraints": [
            {"type": "UNIQUENESS", "node_type": "Person", "property_name": "email"}
        ],
    }

    # Should raise - email is not allowed because the property is not defined in the node
    with pytest.raises(SchemaValidationError) as exc_info:
        GraphSchema.model_validate(schema_dict)

    assert "Constraint references undefined property 'email'" in str(exc_info.value)


def test_schema_with_valid_constraints() -> None:
    schema_dict: dict[str, Any] = {
        "node_types": [
            {"label": "Person", "properties": [{"name": "name", "type": "STRING"}]}
        ],
        "constraints": [
            {"type": "UNIQUENESS", "node_type": "Person", "property_name": "name"}
        ],
    }
    schema = GraphSchema.model_validate(schema_dict)

    assert len(schema.constraints) == 1
    assert schema.constraints[0].type == "UNIQUENESS"
    assert schema.constraints[0].node_type == "Person"
    assert schema.constraints[0].property_name == "name"


def test_schema_constraint_validation_invalid_node_type() -> None:
    schema_dict: dict[str, Any] = {
        "node_types": [
            {"label": "Person", "properties": [{"name": "name", "type": "STRING"}]}
        ],
        "constraints": [
            {
                "type": "UNIQUENESS",
                "node_type": "NonExistentNode",
                "property_name": "id",
            }
        ],
    }

    with pytest.raises(SchemaValidationError) as exc_info:
        GraphSchema.model_validate(schema_dict)

    assert "Constraint references undefined node type: NonExistentNode" in str(
        exc_info.value
    )


def test_schema_constraint_validation_missing_property_name() -> None:
    schema_dict: dict[str, Any] = {
        "node_types": [
            {"label": "Person", "properties": [{"name": "name", "type": "STRING"}]}
        ],
        "constraints": [
            {"type": "UNIQUENESS", "node_type": "Person", "property_name": ""}
        ],
    }

    with pytest.raises(SchemaValidationError) as exc_info:
        GraphSchema.model_validate(schema_dict)

    assert "Constraint has no property name" in str(exc_info.value)


@pytest.fixture
def valid_node_types() -> tuple[NodeType, ...]:
    return (
        NodeType(
            label="PERSON",
            description="An individual human being.",
            properties=[
                PropertyType(name="birth date", type="ZONED_DATETIME"),
                PropertyType(name="name", type="STRING", required=True),
            ],
            additional_properties=False,
        ),
        NodeType(
            label="ORGANIZATION",
            description="A structured group of people with a common purpose.",
        ),
        NodeType(label="AGE", description="Age of a person in years."),
    )


@pytest.fixture
def valid_relationship_types() -> tuple[RelationshipType, ...]:
    return (
        RelationshipType(
            label="EMPLOYED_BY",
            description="Indicates employment relationship.",
            properties=[
                PropertyType(name="start_time", type="LOCAL_DATETIME", required=True),
                PropertyType(name="end_time", type="LOCAL_DATETIME"),
            ],
            additional_properties=False,
        ),
        RelationshipType(
            label="ORGANIZED_BY",
            description="Indicates organization responsible for an event.",
        ),
        RelationshipType(
            label="ATTENDED_BY", description="Indicates attendance at an event."
        ),
    )


@pytest.fixture
def valid_patterns() -> tuple[tuple[str, str, str], ...]:
    return (
        ("PERSON", "EMPLOYED_BY", "ORGANIZATION"),
        ("ORGANIZATION", "ATTENDED_BY", "PERSON"),
    )


@pytest.fixture
def patterns_with_invalid_entity() -> tuple[tuple[str, str, str], ...]:
    return (
        ("PERSON", "EMPLOYED_BY", "ORGANIZATION"),
        ("NON_EXISTENT_ENTITY", "ATTENDED_BY", "PERSON"),
    )


@pytest.fixture
def valid_constraints() -> tuple[ConstraintType, ...]:
    return (
        ConstraintType(type="UNIQUENESS", node_type="PERSON", property_name="name"),
    )


@pytest.fixture
def patterns_with_invalid_relation() -> tuple[tuple[str, str, str], ...]:
    return (("PERSON", "NON_EXISTENT_RELATION", "ORGANIZATION"),)


@pytest.fixture
def schema_builder() -> SchemaBuilder:
    return SchemaBuilder()


@pytest.fixture
def graph_schema(
    schema_builder: SchemaBuilder,
    valid_node_types: Tuple[NodeType, ...],
    valid_relationship_types: Tuple[RelationshipType, ...],
    valid_patterns: Tuple[Tuple[str, str, str], ...],
) -> GraphSchema:
    return schema_builder.create_schema_model(
        list(valid_node_types), list(valid_relationship_types), list(valid_patterns)
    )


def test_create_schema_model_valid_data(
    schema_builder: SchemaBuilder,
    valid_node_types: Tuple[NodeType, ...],
    valid_relationship_types: Tuple[RelationshipType, ...],
    valid_patterns: Tuple[Tuple[str, str, str], ...],
) -> None:
    schema = schema_builder.create_schema_model(
        list(valid_node_types), list(valid_relationship_types), list(valid_patterns)
    )

    assert schema.node_types == valid_node_types
    assert schema.relationship_types == valid_relationship_types
    assert schema.patterns == valid_patterns
    assert schema.additional_node_types is False
    assert schema.additional_relationship_types is False
    assert schema.additional_patterns is False


def test_create_schema_model_with_constraints(
    schema_builder: SchemaBuilder,
    valid_node_types: Tuple[NodeType, ...],
    valid_constraints: Tuple[ConstraintType, ...],
) -> None:
    schema = schema_builder.create_schema_model(
        list(valid_node_types),
        constraints=list(valid_constraints),
    )

    assert schema.node_types == valid_node_types
    assert schema.constraints == valid_constraints
    assert len(schema.constraints) == 1
    assert schema.constraints[0].type == "UNIQUENESS"
    assert schema.constraints[0].node_type == "PERSON"
    assert schema.constraints[0].property_name == "name"


@pytest.mark.asyncio
async def test_run_method(
    schema_builder: SchemaBuilder,
    valid_node_types: Tuple[NodeType, ...],
    valid_relationship_types: Tuple[RelationshipType, ...],
    valid_patterns: Tuple[Tuple[str, str, str], ...],
) -> None:
    with patch.object(
        schema_builder,
        "create_schema_model",
        return_value=GraphSchema(
            node_types=valid_node_types,
            relationship_types=valid_relationship_types,
            patterns=valid_patterns,
        ),
    ):
        schema = await schema_builder.run(
            list(valid_node_types), list(valid_relationship_types), list(valid_patterns)
        )

    assert schema.node_types == valid_node_types
    assert schema.relationship_types == valid_relationship_types
    assert schema.patterns == valid_patterns
    assert schema.additional_node_types is False
    assert schema.additional_relationship_types is False
    assert schema.additional_patterns is False


@pytest.mark.asyncio
async def test_run_method_with_constraints(
    schema_builder: SchemaBuilder,
    valid_node_types: Tuple[NodeType, ...],
    valid_constraints: Tuple[ConstraintType, ...],
) -> None:
    schema = await schema_builder.run(
        list(valid_node_types),
        constraints=list(valid_constraints),
    )

    assert schema.node_types == valid_node_types
    assert schema.constraints == valid_constraints
    assert len(schema.constraints) == 1
    assert schema.constraints[0].type == "UNIQUENESS"
    assert schema.constraints[0].node_type == "PERSON"
    assert schema.constraints[0].property_name == "name"


def test_create_schema_model_invalid_entity(
    schema_builder: SchemaBuilder,
    valid_node_types: Tuple[NodeType, ...],
    valid_relationship_types: Tuple[RelationshipType, ...],
    patterns_with_invalid_entity: Tuple[Tuple[str, str, str], ...],
) -> None:
    with pytest.raises(SchemaValidationError) as exc_info:
        schema_builder.create_schema_model(
            list(valid_node_types),
            list(valid_relationship_types),
            list(patterns_with_invalid_entity),
        )
    assert "Node type 'NON_EXISTENT_ENTITY' is not defined" in str(
        exc_info.value
    ), "Should fail due to non-existent entity"


def test_create_schema_model_invalid_relation(
    schema_builder: SchemaBuilder,
    valid_node_types: Tuple[NodeType, ...],
    valid_relationship_types: Tuple[RelationshipType, ...],
    patterns_with_invalid_relation: Tuple[Tuple[str, str, str], ...],
) -> None:
    with pytest.raises(SchemaValidationError) as exc_info:
        schema_builder.create_schema_model(
            list(valid_node_types),
            list(valid_relationship_types),
            list(patterns_with_invalid_relation),
        )
    assert "Relationship type 'NON_EXISTENT_RELATION' is not defined" in str(
        exc_info.value
    ), "Should fail due to non-existent relation"


def test_create_schema_model_no_potential_schema(
    schema_builder: SchemaBuilder,
    valid_node_types: Tuple[NodeType, ...],
    valid_relationship_types: Tuple[RelationshipType, ...],
) -> None:
    schema_instance = schema_builder.create_schema_model(
        list(valid_node_types), list(valid_relationship_types)
    )
    assert schema_instance.node_types == valid_node_types
    assert schema_instance.relationship_types == valid_relationship_types
    assert schema_instance.patterns == ()


def test_create_schema_model_no_relations_or_potential_schema(
    schema_builder: SchemaBuilder,
    valid_node_types: Tuple[NodeType, ...],
) -> None:
    schema_instance = schema_builder.create_schema_model(list(valid_node_types))

    assert len(schema_instance.node_types) == 3
    person = schema_instance.node_type_from_label("PERSON")

    assert person is not None
    assert person.description == "An individual human being."
    assert len(person.properties) == 2
    assert person.additional_properties is False

    org = schema_instance.node_type_from_label("ORGANIZATION")
    assert org is not None
    assert org.description == "A structured group of people with a common purpose."
    assert org.additional_properties is True

    age = schema_instance.node_type_from_label("AGE")
    assert age is not None
    assert age.description == "Age of a person in years."
    assert age.additional_properties is True


def test_create_schema_model_missing_relations(
    schema_builder: SchemaBuilder,
    valid_node_types: Tuple[NodeType, ...],
    valid_patterns: Tuple[Tuple[str, str, str], ...],
) -> None:
    with pytest.raises(SchemaValidationError) as exc_info:
        schema_builder.create_schema_model(
            node_types=valid_node_types, patterns=valid_patterns
        )
    assert "Relationship types must also be provided when using patterns." in str(
        exc_info.value
    ), "Should fail due to missing relations"


@pytest.fixture
def mock_llm() -> AsyncMock:
    mock = AsyncMock()
    mock.ainvoke = AsyncMock()
    return mock


@pytest.fixture
def valid_schema_json() -> str:
    return """
    {
        "node_types": [
            {
                "label": "Person",
                "properties": [
                    {"name": "name", "type": "STRING"}
                ]
            },
            {
                "label": "Organization",
                "properties": [
                    {"name": "name", "type": "STRING"}
                ]
            }
        ],
        "relationship_types": [
            {
                "label": "WORKS_FOR",
                "properties": [
                    {"name": "since", "type": "DATE"}
                ]
            }
        ],
        "patterns": [
            ["Person", "WORKS_FOR", "Organization"]
        ]
    }
    """


@pytest.fixture
def schema_json_with_valid_constraints() -> str:
    return """
    {
        "node_types": [
            {
                "label": "Person",
                "properties": [
                    {"name": "name", "type": "STRING"},
                    {"name": "email", "type": "STRING"}
                ]
            },
            {
                "label": "Organization",
                "properties": [
                    {"name": "name", "type": "STRING"}
                ]
            }
        ],
        "relationship_types": [
            {
                "label": "WORKS_FOR",
                "properties": [
                    {"name": "since", "type": "DATE"}
                ]
            }
        ],
        "patterns": [
            ["Person", "WORKS_FOR", "Organization"]
        ],
        "constraints": [
            {"type": "UNIQUENESS", "node_type": "Person", "property_name": "name"}
        ]
    }
    """


@pytest.fixture
def schema_json_with_invalid_constraints() -> str:
    return """
    {
        "node_types": [
            {
                "label": "Person",
                "properties": [
                    {"name": "name", "type": "STRING"}
                ]
            },
            {
                "label": "Organization",
                "properties": [
                    {"name": "name", "type": "STRING"}
                ]
            }
        ],
        "relationship_types": [
            {
                "label": "WORKS_FOR",
                "properties": [
                    {"name": "since", "type": "DATE"}
                ]
            }
        ],
        "patterns": [
            ["Person", "WORKS_FOR", "Organization"]
        ],
        "constraints": [
            {"type": "UNIQUENESS", "node_type": "Person", "property_name": "name"},
            {"type": "UNIQUENESS", "node_type": "Person", "property_name": "email"},
            {"type": "UNIQUENESS", "node_type": "NonExistentNode", "property_name": "id"},
            {"type": "UNIQUENESS", "node_type": "Person", "property_name": ""}
        ]
    }
    """


@pytest.fixture
def schema_json_with_null_constraints() -> str:
    return """
    {
        "node_types": [
            {
                "label": "Person",
                "properties": [
                    {"name": "name", "type": "STRING"}
                ]
            },
            {
                "label": "Organization",
                "properties": [
                    {"name": "name", "type": "STRING"}
                ]
            }
        ],
        "relationship_types": [
            {
                "label": "WORKS_FOR",
                "properties": [
                    {"name": "since", "type": "DATE"}
                ]
            }
        ],
        "patterns": [
            ["Person", "WORKS_FOR", "Organization"]
        ],
        "constraints": null
    }
    """


@pytest.fixture
def invalid_schema_json() -> str:
    return """
    {
        "node_types": [
            {
                "label": "Person",
            },
        ],
        invalid json content
    }
    """


@pytest.fixture
def schema_from_text(mock_llm: AsyncMock) -> SchemaFromTextExtractor:
    return SchemaFromTextExtractor(llm=mock_llm)


@pytest.mark.asyncio
async def test_schema_from_text_run_valid_response(
    schema_from_text: SchemaFromTextExtractor,
    mock_llm: AsyncMock,
    valid_schema_json: str,
) -> None:
    # configure the mock LLM to return a valid schema JSON
    mock_llm.ainvoke.return_value = LLMResponse(content=valid_schema_json)

    # run the schema extraction
    schema_config = await schema_from_text.run(text="Sample text for extraction")

    # verify the LLM was called with a prompt
    mock_llm.ainvoke.assert_called_once()
    prompt_arg = mock_llm.ainvoke.call_args[0][0]
    assert isinstance(prompt_arg, str)
    assert "Sample text for extraction" in prompt_arg

    # verify the schema was correctly extracted
    assert len(schema_config.node_types) == 2
    assert schema_config.node_type_from_label("Person") is not None
    assert schema_config.node_type_from_label("Organization") is not None

    assert schema_config.relationship_types is not None
    assert schema_config.relationship_type_from_label("WORKS_FOR") is not None

    assert schema_config.patterns is not None
    assert len(schema_config.patterns) == 1
    assert schema_config.patterns[0] == ("Person", "WORKS_FOR", "Organization")


@pytest.mark.asyncio
async def test_schema_from_text_run_invalid_json(
    schema_from_text: SchemaFromTextExtractor,
    mock_llm: AsyncMock,
    invalid_schema_json: str,
) -> None:
    # configure the mock LLM to return invalid JSON
    mock_llm.ainvoke.return_value = LLMResponse(content=invalid_schema_json)

    # verify that running with invalid JSON raises a ValueError
    with pytest.raises(SchemaExtractionError) as exc_info:
        await schema_from_text.run(text="Sample text for extraction")

    assert "not valid JSON" in str(exc_info.value)


@pytest.mark.asyncio
async def test_schema_from_text_custom_template(
    mock_llm: AsyncMock, valid_schema_json: str
) -> None:
    # create a  custom template
    custom_prompt = "This is a custom prompt with text: {text}"
    custom_template = PromptTemplate(template=custom_prompt, expected_inputs=["text"])

    # create SchemaFromTextExtractor with the custom template
    schema_from_text = SchemaFromTextExtractor(
        llm=mock_llm, prompt_template=custom_template
    )

    # configure mock LLM to return valid JSON and capture the prompt that was sent to it
    mock_llm.ainvoke.return_value = LLMResponse(content=valid_schema_json)

    # run the schema extraction
    await schema_from_text.run(text="Sample text")

    # verify the custom prompt was passed to the LLM
    prompt_sent_to_llm = mock_llm.ainvoke.call_args[0][0]
    assert "This is a custom prompt with text" in prompt_sent_to_llm


@pytest.mark.asyncio
async def test_schema_from_text_llm_params(
    mock_llm: AsyncMock, valid_schema_json: str
) -> None:
    # configure custom LLM parameters
    llm_params = {"temperature": 0.1, "max_tokens": 500}

    # create SchemaFromTextExtractor with custom LLM parameters
    schema_from_text = SchemaFromTextExtractor(llm=mock_llm, llm_params=llm_params)

    # configure the mock LLM to return a valid schema JSON
    mock_llm.ainvoke.return_value = LLMResponse(content=valid_schema_json)

    # run the schema extraction
    await schema_from_text.run(text="Sample text")

    # verify the LLM was called with the custom parameters
    mock_llm.ainvoke.assert_called_once()
    call_kwargs = mock_llm.ainvoke.call_args[1]
    assert call_kwargs["temperature"] == 0.1
    assert call_kwargs["max_tokens"] == 500


@pytest.mark.asyncio
async def test_schema_config_save_json(graph_schema: GraphSchema) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        # create file path
        json_path = os.path.join(temp_dir, "schema.json")

        # store the schema config
        graph_schema.save(json_path)

        # verify the file exists and has content
        assert os.path.exists(json_path)
        assert os.path.getsize(json_path) > 0

        # verify the content is valid JSON and contains expected data
        with open(json_path, "r") as f:
            data = json.load(f)
            assert "node_types" in data
            assert len(data["node_types"]) == 3


@pytest.mark.asyncio
async def test_schema_config_save_yaml(graph_schema: GraphSchema) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create file path
        yaml_path = os.path.join(temp_dir, "schema.yaml")

        # Store the schema config
        graph_schema.save(yaml_path)

        # Verify the file exists and has content
        assert os.path.exists(yaml_path)
        assert os.path.getsize(yaml_path) > 0

        # Verify the content is valid YAML and contains expected data
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
            assert "node_types" in data
            assert len(data["node_types"]) == 3


@pytest.mark.asyncio
async def test_schema_config_from_file(graph_schema: GraphSchema) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        # create file paths with different extensions
        json_path = os.path.join(temp_dir, "schema.json")
        yaml_path = os.path.join(temp_dir, "schema.yaml")
        yml_path = os.path.join(temp_dir, "schema.yml")

        # store the schema config in the different formats
        graph_schema.save(json_path)
        graph_schema.save(yaml_path)
        graph_schema.save(yml_path)

        # load using from_file which should detect the format based on extension
        json_schema = GraphSchema.from_file(json_path)
        yaml_schema = GraphSchema.from_file(yaml_path)
        yml_schema = GraphSchema.from_file(yml_path)

        # simple verification that the objects were loaded correctly
        assert isinstance(json_schema, GraphSchema)
        assert isinstance(yaml_schema, GraphSchema)
        assert isinstance(yml_schema, GraphSchema)

        # verify basic structure is intact
        assert "node_types" in json_schema.model_dump()
        assert "node_types" in yaml_schema.model_dump()
        assert "node_types" in yml_schema.model_dump()

        # verify an unsupported extension raises the correct error
        txt_path = os.path.join(temp_dir, "schema.txt")
        graph_schema.save(
            txt_path, format=FileFormat.JSON
        )  # Store as JSON but with .txt extension

        with pytest.raises(ValueError, match="Unsupported file format: None"):
            GraphSchema.from_file(txt_path)


@pytest.fixture
def valid_schema_json_array() -> str:
    return """
    [
        {
            "node_types": [
                {
                    "label": "Person",
                    "properties": [
                        {"name": "name", "type": "STRING"}
                    ]
                },
                {
                    "label": "Organization",
                    "properties": [
                        {"name": "name", "type": "STRING"}
                    ]
                }
            ],
            "relationship_types": [
                {
                    "label": "WORKS_FOR",
                    "properties": [
                        {"name": "since", "type": "DATE"}
                    ]
                }
            ],
            "patterns": [
                ["Person", "WORKS_FOR", "Organization"]
            ]
        }
    ]
    """


@pytest.mark.asyncio
async def test_schema_from_text_run_valid_json_array(
    schema_from_text: SchemaFromTextExtractor,
    mock_llm: AsyncMock,
    valid_schema_json_array: str,
) -> None:
    # configure the mock LLM to return a valid JSON array
    mock_llm.ainvoke.return_value = LLMResponse(content=valid_schema_json_array)

    # run the schema extraction
    schema = await schema_from_text.run(text="Sample text for extraction")

    # verify the schema was correctly extracted from the array
    assert len(schema.node_types) == 2
    assert schema.node_type_from_label("Person") is not None
    assert schema.node_type_from_label("Organization") is not None

    assert schema.relationship_types is not None
    assert schema.relationship_type_from_label("WORKS_FOR") is not None

    assert schema.patterns is not None
    assert len(schema.patterns) == 1
    assert schema.patterns[0] == ("Person", "WORKS_FOR", "Organization")


@pytest.fixture
def schema_json_with_invalid_node_patterns() -> str:
    return """
    {
        "node_types": [
            {
                "label": "Person",
                "properties": [
                    {"name": "name", "type": "STRING"}
                ]
            },
            {
                "label": "Organization",
                "properties": [
                    {"name": "name", "type": "STRING"}
                ]
            }
        ],
        "relationship_types": [
            {
                "label": "WORKS_FOR",
                "properties": [
                    {"name": "since", "type": "DATE"}
                ]
            }
        ],
        "patterns": [
            ["Person", "WORKS_FOR", "Organization"],
            ["Person", "WORKS_FOR", "UndefinedNode"],
            ["UndefinedNode", "WORKS_FOR", "Organization"]
        ]
    }
    """


@pytest.fixture
def schema_json_with_invalid_relationship_patterns() -> str:
    return """
    {
        "node_types": [
            {
                "label": "Person",
                "properties": [
                    {"name": "name", "type": "STRING"}
                ]
            },
            {
                "label": "Organization",
                "properties": [
                    {"name": "name", "type": "STRING"}
                ]
            }
        ],
        "relationship_types": [
            {
                "label": "WORKS_FOR",
                "properties": [
                    {"name": "since", "type": "DATE"}
                ]
            }
        ],
        "patterns": [
            ["Person", "WORKS_FOR", "Organization"],
            ["Person", "UNDEFINED_RELATION", "Organization"],
            ["Organization", "ANOTHER_UNDEFINED_RELATION", "Person"]
        ]
    }
    """


@pytest.fixture
def schema_json_with_nodes_without_labels() -> str:
    return """
    {
        "node_types": [
            {
                "label": "Person",
                "properties": [
                    {"name": "name", "type": "STRING"}
                ]
            },
            {
                "properties": [
                    {"name": "name", "type": "STRING"}
                ]
            },
            {
                "label": "",
                "properties": [
                    {"name": "name", "type": "STRING"}
                ]
            },
            "Organization",
            "",
            "Company",
            "Invalid description with spaces",
            "{\\"invalid\\": \\"json object\\"}"
        ],
        "relationship_types": [
            {
                "label": "WORKS_FOR",
                "properties": [
                    {"name": "since", "type": "DATE"}
                ]
            }
        ],
        "patterns": [
            ["Person", "WORKS_FOR", "Organization"]
        ]
    }
    """


@pytest.fixture
def schema_json_with_relationships_without_labels() -> str:
    return """
    {
        "node_types": [
            {
                "label": "Person",
                "properties": [
                    {"name": "name", "type": "STRING"}
                ]
            },
            {
                "label": "Organization",
                "properties": [
                    {"name": "name", "type": "STRING"}
                ]
            }
        ],
        "relationship_types": [
            {
                "label": "WORKS_FOR",
                "properties": [
                    {"name": "since", "type": "DATE"}
                ]
            },
            {
                "properties": [
                    {"name": "since", "type": "DATE"}
                ]
            },
            {
                "label": "",
                "properties": [
                    {"name": "since", "type": "DATE"}
                ]
            },
            "MANAGES",
            "",
            "SUPERVISES",
            "invalid relationship description",
            "{\\"invalid\\": \\"json\\"}"
        ],
        "patterns": [
            ["Person", "WORKS_FOR", "Organization"],
            ["Person", "MANAGES", "Organization"]
        ]
    }
    """


@pytest.fixture
def schema_json_with_nonexistent_property_constraint() -> str:
    return """
    {
        "node_types": [
            {
                "label": "Person",
                "properties": [
                    {"name": "name", "type": "STRING"}
                ]
            }
        ],
        "relationship_types": [],
        "patterns": [],
        "constraints": [
            {"type": "UNIQUENESS", "node_type": "Person", "property_name": "name"},
            {"type": "UNIQUENESS", "node_type": "Person", "property_name": "nonexistent_property"}
        ]
    }
    """


@pytest.mark.asyncio
async def test_schema_from_text_filters_invalid_node_patterns(
    schema_from_text: SchemaFromTextExtractor,
    mock_llm: AsyncMock,
    schema_json_with_invalid_node_patterns: str,
) -> None:
    # configure the mock LLM to return schema with invalid node patterns
    mock_llm.ainvoke.return_value = LLMResponse(
        content=schema_json_with_invalid_node_patterns
    )

    # run the schema extraction
    schema = await schema_from_text.run(text="Sample text for extraction")

    # verify that invalid node patterns were filtered out (2 out of 3 patterns should be removed)
    assert schema.patterns is not None
    assert len(schema.patterns) == 1
    assert schema.patterns[0] == ("Person", "WORKS_FOR", "Organization")


@pytest.mark.asyncio
async def test_schema_from_text_filters_invalid_relationship_patterns(
    schema_from_text: SchemaFromTextExtractor,
    mock_llm: AsyncMock,
    schema_json_with_invalid_relationship_patterns: str,
) -> None:
    # configure the mock LLM to return schema with invalid relationship patterns
    mock_llm.ainvoke.return_value = LLMResponse(
        content=schema_json_with_invalid_relationship_patterns
    )

    # run the schema extraction
    schema = await schema_from_text.run(text="Sample text for extraction")

    # verify that invalid relationship patterns were filtered out (2 out of 3 patterns should be removed)
    assert schema.patterns is not None
    assert len(schema.patterns) == 1
    assert schema.patterns[0] == ("Person", "WORKS_FOR", "Organization")


@pytest.mark.asyncio
async def test_schema_from_text_filters_nodes_without_labels(
    schema_from_text: SchemaFromTextExtractor,
    mock_llm: AsyncMock,
    schema_json_with_nodes_without_labels: str,
) -> None:
    # configure the mock LLM to return schema with nodes without labels
    mock_llm.ainvoke.return_value = LLMResponse(
        content=schema_json_with_nodes_without_labels
    )

    # run the schema extraction
    schema = await schema_from_text.run(text="Sample text for extraction")

    # verify that nodes without labels were filtered out (5 out of 8 nodes should be removed)
    assert len(schema.node_types) == 3
    assert schema.node_type_from_label("Person") is not None
    assert schema.node_type_from_label("Organization") is not None
    assert schema.node_type_from_label("Company") is not None

    # verify that the pattern is still valid with the remaining nodes
    assert schema.patterns is not None
    assert len(schema.patterns) == 1
    assert schema.patterns[0] == ("Person", "WORKS_FOR", "Organization")


@pytest.mark.asyncio
async def test_schema_from_text_filters_relationships_without_labels(
    schema_from_text: SchemaFromTextExtractor,
    mock_llm: AsyncMock,
    schema_json_with_relationships_without_labels: str,
) -> None:
    # configure the mock LLM to return schema with relationships without labels
    mock_llm.ainvoke.return_value = LLMResponse(
        content=schema_json_with_relationships_without_labels
    )

    # run the schema extraction
    schema = await schema_from_text.run(text="Sample text for extraction")

    # verify that relationships without labels were filtered out (5 out of 8 relationships should be removed)
    assert schema.relationship_types is not None
    assert len(schema.relationship_types) == 3
    assert schema.relationship_type_from_label("WORKS_FOR") is not None
    assert schema.relationship_type_from_label("MANAGES") is not None
    assert schema.relationship_type_from_label("SUPERVISES") is not None

    # verify that the patterns are still valid with the remaining relationships
    assert schema.patterns is not None
    assert len(schema.patterns) == 2
    assert ("Person", "WORKS_FOR", "Organization") in schema.patterns
    assert ("Person", "MANAGES", "Organization") in schema.patterns


@pytest.mark.asyncio
async def test_schema_from_text_with_valid_constraints(
    schema_from_text: SchemaFromTextExtractor,
    mock_llm: AsyncMock,
    schema_json_with_valid_constraints: str,
) -> None:
    # configure the mock LLM to return schema with valid constraints
    mock_llm.ainvoke.return_value = LLMResponse(
        content=schema_json_with_valid_constraints
    )

    # run the schema extraction
    schema = await schema_from_text.run(text="Sample text for extraction")

    assert len(schema.constraints) == 1
    assert schema.constraints[0].type == "UNIQUENESS"
    assert schema.constraints[0].node_type == "Person"
    assert schema.constraints[0].property_name == "name"


@pytest.mark.asyncio
async def test_schema_from_text_filters_invalid_constraints(
    schema_from_text: SchemaFromTextExtractor,
    mock_llm: AsyncMock,
    schema_json_with_invalid_constraints: str,
) -> None:
    # configure the mock LLM to return schema with invalid constraints
    mock_llm.ainvoke.return_value = LLMResponse(
        content=schema_json_with_invalid_constraints
    )

    # run the schema extraction
    schema = await schema_from_text.run(text="Sample text for extraction")

    # verify that invalid constraints were filtered out:
    # constraints with NonExistentNode should be removed
    # constraint with empty property_name should be removed
    # only the valid constraint should remain
    assert len(schema.constraints) == 1
    assert schema.constraints[0].node_type == "Person"
    assert schema.constraints[0].property_name == "name"


@pytest.mark.asyncio
async def test_schema_from_text_filters_constraint_with_nonexistent_property(
    schema_from_text: SchemaFromTextExtractor,
    mock_llm: AsyncMock,
    schema_json_with_nonexistent_property_constraint: str,
) -> None:
    # configure the mock LLM to return schema with constraint on nonexistent property
    mock_llm.ainvoke.return_value = LLMResponse(
        content=schema_json_with_nonexistent_property_constraint
    )

    # run the schema extraction
    schema = await schema_from_text.run(text="Sample text for extraction")

    # verify that only the valid constraint (with "name" property) remains
    # the constraint with "nonexistent_property" should be filtered out
    assert len(schema.constraints) == 1
    assert schema.constraints[0].property_name == "name"


@pytest.mark.asyncio
async def test_schema_from_text_handles_null_constraints(
    schema_from_text: SchemaFromTextExtractor,
    mock_llm: AsyncMock,
    schema_json_with_null_constraints: str,
) -> None:
    # configure the mock LLM to return schema with null constraints
    mock_llm.ainvoke.return_value = LLMResponse(
        content=schema_json_with_null_constraints
    )

    # run the schema extraction - should not crash
    schema = await schema_from_text.run(text="Sample text for extraction")

    # verify schema was created with empty constraints
    assert len(schema.constraints) == 0


@pytest.mark.asyncio
async def test_schema_from_text_handles_missing_constraints(
    schema_from_text: SchemaFromTextExtractor,
    mock_llm: AsyncMock,
    valid_schema_json: str,
) -> None:
    # configure the mock LLM to return schema without constraints field
    mock_llm.ainvoke.return_value = LLMResponse(content=valid_schema_json)

    #  run the schema extraction - should not crash
    schema = await schema_from_text.run(text="Sample text for extraction")

    # verify schema was created with empty constraints
    assert len(schema.constraints) == 0


def test_clean_json_content_markdown_with_json_language(
    schema_from_text: SchemaFromTextExtractor,
) -> None:
    content = """```json
{"node_types": [{"label": "Person"}]}
```"""

    cleaned = schema_from_text._clean_json_content(content)
    assert cleaned == '{"node_types": [{"label": "Person"}]}'


def test_clean_json_content_markdown_without_language(
    schema_from_text: SchemaFromTextExtractor,
) -> None:
    content = """```
{"node_types": [{"label": "Person"}]}
```"""

    cleaned = schema_from_text._clean_json_content(content)
    assert cleaned == '{"node_types": [{"label": "Person"}]}'


def test_clean_json_content_plain_json(
    schema_from_text: SchemaFromTextExtractor,
) -> None:
    content = '{"node_types": [{"label": "Person"}]}'

    cleaned = schema_from_text._clean_json_content(content)
    assert cleaned == '{"node_types": [{"label": "Person"}]}'


@pytest.mark.asyncio
@patch("neo4j_graphrag.experimental.components.schema.get_structured_schema")
async def test_schema_from_existing_graph(mock_get_structured_schema: Mock) -> None:
    mock_get_structured_schema.return_value = {
        "node_props": {
            "Person": [
                {"property": "id", "type": "INTEGER"},
                {"property": "name", "type": "STRING"},
            ]
        },
        "rel_props": {"KNOWS": [{"property": "fromDate", "type": "DATE"}]},
        "relationships": [
            {"start": "Person", "type": "KNOWS", "end": "Person"},
            {"start": "Person", "type": "LIVES_IN", "end": "City"},
        ],
        "metadata": {
            "constraint": [
                {
                    "id": 7,
                    "name": "person_id",
                    "type": "NODE_PROPERTY_EXISTENCE",
                    "entityType": "NODE",
                    "labelsOrTypes": ["Person"],
                    "properties": ["id"],
                    "ownedIndex": "person_id",
                    "propertyType": None,
                },
            ],
            "index": [
                {
                    "label": "Person",
                    "properties": ["name"],
                    "size": 2,
                    "type": "RANGE",
                    "valuesSelectivity": 1.0,
                    "distinctValues": 2.0,
                },
            ],
        },
    }
    driver = Mock()
    schema_builder = SchemaFromExistingGraphExtractor(
        driver=driver,
    )
    schema = await schema_builder.run()
    assert isinstance(schema, GraphSchema)
    assert len(schema.node_types) == 2
    person_node_type = schema.node_type_from_label("Person")
    assert person_node_type is not None
    id_person_property = [p for p in person_node_type.properties if p.name == "id"][0]
    assert id_person_property.required is True
    assert person_node_type.additional_properties is False
    city_node_type = schema.node_type_from_label("City")
    assert city_node_type is not None
    assert city_node_type.additional_properties is True

    assert len(schema.relationship_types) == 2
    knows_rel = schema.relationship_type_from_label("KNOWS")
    assert knows_rel is not None
    assert knows_rel.additional_properties is False
    lives_rel = schema.relationship_type_from_label("LIVES_IN")
    assert lives_rel is not None
    assert lives_rel.additional_properties is True

    assert schema.additional_node_types is False
    assert schema.additional_relationship_types is False
    assert schema.additional_patterns is False

    assert schema.patterns == (
        ("Person", "KNOWS", "Person"),
        ("Person", "LIVES_IN", "City"),
    )


@pytest.mark.asyncio
@patch("neo4j_graphrag.experimental.components.schema.get_structured_schema")
async def test_schema_from_existing_graph_additional_params(
    mock_get_structured_schema: Mock,
) -> None:
    mock_get_structured_schema.return_value = {
        "node_props": {
            "Person": [
                {"property": "id", "type": "INTEGER"},
                {"property": "name", "type": "STRING"},
            ]
        },
        "rel_props": {"KNOWS": [{"property": "fromDate", "type": "DATE"}]},
        "relationships": [
            {"start": "Person", "type": "KNOWS", "end": "Person"},
            {"start": "Person", "type": "LIVES_IN", "end": "City"},
        ],
    }
    driver = Mock()
    schema_builder = SchemaFromExistingGraphExtractor(
        driver=driver,
        additional_node_types=True,
        additional_relationship_types=True,
        additional_patterns=True,
        additional_properties=True,
    )
    schema = await schema_builder.run()
    person_node_type = schema.node_type_from_label("Person")
    assert person_node_type is not None
    assert person_node_type.additional_properties is True
    city_node_type = schema.node_type_from_label("City")
    assert city_node_type is not None
    assert city_node_type.additional_properties is True

    knows_rel = schema.relationship_type_from_label("KNOWS")
    assert knows_rel is not None
    assert knows_rel.additional_properties is True
    lives_rel = schema.relationship_type_from_label("LIVES_IN")
    assert lives_rel is not None
    assert lives_rel.additional_properties is True

    assert schema.additional_node_types is True
    assert schema.additional_relationship_types is True
    assert schema.additional_patterns is True
