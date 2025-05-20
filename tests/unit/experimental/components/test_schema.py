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
from typing import Tuple
from unittest.mock import AsyncMock

import pytest

from neo4j_graphrag.exceptions import SchemaValidationError, SchemaExtractionError
from neo4j_graphrag.experimental.components.schema import (
    SchemaBuilder,
    NodeType,
    PropertyType,
    RelationshipType,
    SchemaFromTextExtractor,
    GraphSchema,
)
import os
import tempfile
import yaml

from neo4j_graphrag.generation import PromptTemplate
from neo4j_graphrag.llm.types import LLMResponse


@pytest.fixture
def valid_node_types() -> tuple[NodeType, ...]:
    return (
        NodeType(
            label="PERSON",
            description="An individual human being.",
            properties=[
                PropertyType(name="birth date", type="ZONED_DATETIME"),
                PropertyType(name="name", type="STRING"),
            ],
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
                PropertyType(name="start_time", type="LOCAL_DATETIME"),
                PropertyType(name="end_time", type="LOCAL_DATETIME"),
            ],
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
    schema_instance = schema_builder.create_schema_model(
        list(valid_node_types), list(valid_relationship_types), list(valid_patterns)
    )

    assert schema_instance.node_types == valid_node_types
    assert schema_instance.relationship_types == valid_relationship_types
    assert schema_instance.patterns == valid_patterns


@pytest.mark.asyncio
async def test_run_method(
    schema_builder: SchemaBuilder,
    valid_node_types: Tuple[NodeType, ...],
    valid_relationship_types: Tuple[RelationshipType, ...],
    valid_patterns: Tuple[Tuple[str, str, str], ...],
) -> None:
    schema = await schema_builder.run(
        list(valid_node_types), list(valid_relationship_types), list(valid_patterns)
    )

    assert schema.node_types == valid_node_types
    assert schema.relationship_types == valid_relationship_types
    assert schema.patterns == valid_patterns


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
    assert "Entity 'NON_EXISTENT_ENTITY' is not defined" in str(
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
    assert "Relation 'NON_EXISTENT_RELATION' is not defined" in str(
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
    assert schema_instance.patterns is None


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

    org = schema_instance.node_type_from_label("ORGANIZATION")
    assert org is not None
    assert org.description == "A structured group of people with a common purpose."

    age = schema_instance.node_type_from_label("AGE")
    assert age is not None
    assert age.description == "Age of a person in years."


def test_create_schema_model_missing_relations(
    schema_builder: SchemaBuilder,
    valid_node_types: Tuple[NodeType, ...],
    valid_patterns: Tuple[Tuple[str, str, str], ...],
) -> None:
    with pytest.raises(SchemaValidationError) as exc_info:
        schema_builder.create_schema_model(
            node_types=valid_node_types, patterns=valid_patterns
        )
    assert "Relations must also be provided when using a potential schema." in str(
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
async def test_schema_config_store_as_json(graph_schema: GraphSchema) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        # create file path
        json_path = os.path.join(temp_dir, "schema.json")

        # store the schema config
        graph_schema.store_as_json(json_path)

        # verify the file exists and has content
        assert os.path.exists(json_path)
        assert os.path.getsize(json_path) > 0

        # verify the content is valid JSON and contains expected data
        with open(json_path, "r") as f:
            data = json.load(f)
            assert "node_types" in data
            assert len(data["node_types"]) == 3


@pytest.mark.asyncio
async def test_schema_config_store_as_yaml(graph_schema: GraphSchema) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create file path
        yaml_path = os.path.join(temp_dir, "schema.yaml")

        # Store the schema config
        graph_schema.store_as_yaml(yaml_path)

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
        graph_schema.store_as_json(json_path)
        graph_schema.store_as_yaml(yaml_path)
        graph_schema.store_as_yaml(yml_path)

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
        graph_schema.store_as_json(txt_path)  # Store as JSON but with .txt extension

        with pytest.raises(ValueError, match="Unsupported file format"):
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
