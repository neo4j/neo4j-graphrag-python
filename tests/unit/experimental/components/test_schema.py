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
from unittest.mock import AsyncMock

import pytest
from neo4j_graphrag.exceptions import SchemaValidationError
from neo4j_graphrag.experimental.components.schema import (
    SchemaBuilder,
    SchemaEntity,
    SchemaProperty,
    SchemaRelation,
    SchemaFromTextExtractor,
    SchemaConfig,
)
from pydantic import ValidationError
import os
import tempfile
import yaml

from neo4j_graphrag.generation import PromptTemplate
from neo4j_graphrag.llm.types import LLMResponse


@pytest.fixture
def valid_entities() -> list[SchemaEntity]:
    return [
        SchemaEntity(
            label="PERSON",
            description="An individual human being.",
            properties=[
                SchemaProperty(name="birth date", type="ZONED_DATETIME"),
                SchemaProperty(name="name", type="STRING"),
            ],
        ),
        SchemaEntity(
            label="ORGANIZATION",
            description="A structured group of people with a common purpose.",
        ),
        SchemaEntity(label="AGE", description="Age of a person in years."),
    ]


@pytest.fixture
def valid_relations() -> list[SchemaRelation]:
    return [
        SchemaRelation(
            label="EMPLOYED_BY",
            description="Indicates employment relationship.",
            properties=[
                SchemaProperty(name="start_time", type="LOCAL_DATETIME"),
                SchemaProperty(name="end_time", type="LOCAL_DATETIME"),
            ],
        ),
        SchemaRelation(
            label="ORGANIZED_BY",
            description="Indicates organization responsible for an event.",
        ),
        SchemaRelation(
            label="ATTENDED_BY", description="Indicates attendance at an event."
        ),
    ]


@pytest.fixture
def potential_schema() -> list[tuple[str, str, str]]:
    return [
        ("PERSON", "EMPLOYED_BY", "ORGANIZATION"),
        ("ORGANIZATION", "ATTENDED_BY", "PERSON"),
    ]


@pytest.fixture
def potential_schema_with_invalid_entity() -> list[tuple[str, str, str]]:
    return [
        ("PERSON", "EMPLOYED_BY", "ORGANIZATION"),
        ("NON_EXISTENT_ENTITY", "ATTENDED_BY", "PERSON"),
    ]


@pytest.fixture
def potential_schema_with_invalid_relation() -> list[tuple[str, str, str]]:
    return [
        ("PERSON", "NON_EXISTENT_RELATION", "ORGANIZATION"),
    ]


@pytest.fixture
def schema_builder() -> SchemaBuilder:
    return SchemaBuilder()


@pytest.fixture
def schema_config(
    schema_builder: SchemaBuilder,
    valid_entities: list[SchemaEntity],
    valid_relations: list[SchemaRelation],
    potential_schema: list[tuple[str, str, str]],
) -> SchemaConfig:
    return schema_builder.create_schema_model(
        valid_entities, valid_relations, potential_schema
    )


def test_create_schema_model_valid_data(
    schema_builder: SchemaBuilder,
    valid_entities: list[SchemaEntity],
    valid_relations: list[SchemaRelation],
    potential_schema: list[tuple[str, str, str]],
) -> None:
    schema_instance = schema_builder.create_schema_model(
        valid_entities, valid_relations, potential_schema
    )

    assert (
        schema_instance.entities["PERSON"]["description"]
        == "An individual human being."
    )
    assert schema_instance.entities["PERSON"]["properties"] == [
        {"description": "", "name": "birth date", "type": "ZONED_DATETIME"},
        {"description": "", "name": "name", "type": "STRING"},
    ]
    assert (
        schema_instance.entities["ORGANIZATION"]["description"]
        == "A structured group of people with a common purpose."
    )
    assert schema_instance.entities["AGE"]["description"] == "Age of a person in years."

    assert schema_instance.relations
    assert (
        schema_instance.relations["EMPLOYED_BY"]["description"]
        == "Indicates employment relationship."
    )
    assert (
        schema_instance.relations["ORGANIZED_BY"]["description"]
        == "Indicates organization responsible for an event."
    )
    assert (
        schema_instance.relations["ATTENDED_BY"]["description"]
        == "Indicates attendance at an event."
    )
    assert schema_instance.relations["EMPLOYED_BY"]["properties"] == [
        {"description": "", "name": "start_time", "type": "LOCAL_DATETIME"},
        {"description": "", "name": "end_time", "type": "LOCAL_DATETIME"},
    ]

    assert schema_instance.potential_schema
    assert schema_instance.potential_schema == potential_schema


def test_create_schema_model_missing_description(
    schema_builder: SchemaBuilder, potential_schema: list[tuple[str, str, str]]
) -> None:
    entities = [
        SchemaEntity(label="PERSON", description="An individual human being."),
        SchemaEntity(label="ORGANIZATION", description=""),
        SchemaEntity(label="AGE", description=""),
    ]
    relations = [
        SchemaRelation(
            label="EMPLOYED_BY", description="Indicates employment relationship."
        ),
        SchemaRelation(label="ORGANIZED_BY", description=""),
        SchemaRelation(label="ATTENDED_BY", description=""),
    ]

    schema_instance = schema_builder.create_schema_model(
        entities, relations, potential_schema
    )

    assert schema_instance.entities["ORGANIZATION"]["description"] == ""
    assert schema_instance.entities["AGE"]["description"] == ""
    assert schema_instance.relations
    assert schema_instance.relations["ORGANIZED_BY"]["description"] == ""
    assert schema_instance.relations["ATTENDED_BY"]["description"] == ""


def test_create_schema_model_empty_lists(schema_builder: SchemaBuilder) -> None:
    schema_instance = schema_builder.create_schema_model([], [], [])

    assert schema_instance.entities == {}
    assert schema_instance.relations == {}
    assert schema_instance.potential_schema == []


def test_create_schema_model_invalid_data_types(
    schema_builder: SchemaBuilder, potential_schema: list[tuple[str, str, str]]
) -> None:
    with pytest.raises(ValidationError):
        entities = [
            SchemaEntity(label="PERSON", description="An individual human being."),
            SchemaEntity(
                label="ORGANIZATION",
                description="A structured group of people with a common purpose.",
            ),
        ]
        relations = [
            SchemaRelation(
                label="EMPLOYED_BY", description="Indicates employment relationship."
            ),
            SchemaRelation(
                label=456,  # type: ignore
                description="Indicates organization responsible for an event.",
            ),
        ]

        schema_builder.create_schema_model(entities, relations, potential_schema)


def test_create_schema_model_invalid_properties_types(
    schema_builder: SchemaBuilder,
    potential_schema: list[tuple[str, str, str]],
) -> None:
    with pytest.raises(ValidationError):
        entities = [
            SchemaEntity(
                label="PERSON",
                description="An individual human being.",
                properties=[42, 1337],  # type: ignore
            ),
            SchemaEntity(
                label="ORGANIZATION",
                description="A structured group of people with a common purpose.",
            ),
        ]
        relations = [
            SchemaRelation(
                label="EMPLOYED_BY",
                description="Indicates employment relationship.",
                properties=[42, 1337],  # type: ignore
            ),
            SchemaRelation(
                label="ORGANIZED_BY",
                description="Indicates organization responsible for an event.",
            ),
        ]

        schema_builder.create_schema_model(entities, relations, potential_schema)


@pytest.mark.asyncio
async def test_run_method(
    schema_builder: SchemaBuilder,
    valid_entities: list[SchemaEntity],
    valid_relations: list[SchemaRelation],
    potential_schema: list[tuple[str, str, str]],
) -> None:
    schema = await schema_builder.run(valid_entities, valid_relations, potential_schema)

    assert schema.entities["PERSON"]["description"] == "An individual human being."
    assert (
        schema.entities["ORGANIZATION"]["description"]
        == "A structured group of people with a common purpose."
    )
    assert schema.entities["AGE"]["description"] == "Age of a person in years."

    assert schema.relations
    assert (
        schema.relations["EMPLOYED_BY"]["description"]
        == "Indicates employment relationship."
    )
    assert (
        schema.relations["ORGANIZED_BY"]["description"]
        == "Indicates organization responsible for an event."
    )
    assert (
        schema.relations["ATTENDED_BY"]["description"]
        == "Indicates attendance at an event."
    )

    assert schema.potential_schema
    assert schema.potential_schema == potential_schema


def test_create_schema_model_invalid_entity(
    schema_builder: SchemaBuilder,
    valid_entities: list[SchemaEntity],
    valid_relations: list[SchemaRelation],
    potential_schema_with_invalid_entity: list[tuple[str, str, str]],
) -> None:
    with pytest.raises(SchemaValidationError) as exc_info:
        schema_builder.create_schema_model(
            valid_entities, valid_relations, potential_schema_with_invalid_entity
        )
    assert "Entity 'NON_EXISTENT_ENTITY' is not defined" in str(
        exc_info.value
    ), "Should fail due to non-existent entity"


def test_create_schema_model_invalid_relation(
    schema_builder: SchemaBuilder,
    valid_entities: list[SchemaEntity],
    valid_relations: list[SchemaRelation],
    potential_schema_with_invalid_relation: list[tuple[str, str, str]],
) -> None:
    with pytest.raises(SchemaValidationError) as exc_info:
        schema_builder.create_schema_model(
            valid_entities, valid_relations, potential_schema_with_invalid_relation
        )
    assert "Relation 'NON_EXISTENT_RELATION' is not defined" in str(
        exc_info.value
    ), "Should fail due to non-existent relation"


def test_create_schema_model_missing_properties(
    schema_builder: SchemaBuilder, potential_schema: list[tuple[str, str, str]]
) -> None:
    entities = [
        SchemaEntity(label="PERSON", description="An individual human being."),
        SchemaEntity(
            label="ORGANIZATION",
            description="A structured group of people with a common purpose.",
        ),
        SchemaEntity(label="AGE", description="Age of a person in years."),
    ]

    relations = [
        SchemaRelation(
            label="EMPLOYED_BY", description="Indicates employment relationship."
        ),
        SchemaRelation(
            label="ORGANIZED_BY",
            description="Indicates organization responsible for an event.",
        ),
        SchemaRelation(
            label="ATTENDED_BY", description="Indicates attendance at an event."
        ),
    ]

    schema_instance = schema_builder.create_schema_model(
        entities, relations, potential_schema
    )

    assert (
        schema_instance.entities["PERSON"]["properties"] == []
    ), "Expected empty properties for PERSON"
    assert (
        schema_instance.entities["ORGANIZATION"]["properties"] == []
    ), "Expected empty properties for ORGANIZATION"
    assert (
        schema_instance.entities["AGE"]["properties"] == []
    ), "Expected empty properties for AGE"

    assert schema_instance.relations
    assert (
        schema_instance.relations["EMPLOYED_BY"]["properties"] == []
    ), "Expected empty properties for EMPLOYED_BY"
    assert (
        schema_instance.relations["ORGANIZED_BY"]["properties"] == []
    ), "Expected empty properties for ORGANIZED_BY"
    assert (
        schema_instance.relations["ATTENDED_BY"]["properties"] == []
    ), "Expected empty properties for ATTENDED_BY"


def test_create_schema_model_no_potential_schema(
    schema_builder: SchemaBuilder,
    valid_entities: list[SchemaEntity],
    valid_relations: list[SchemaRelation],
) -> None:
    schema_instance = schema_builder.create_schema_model(
        valid_entities, valid_relations
    )

    assert (
        schema_instance.entities["PERSON"]["description"]
        == "An individual human being."
    )
    assert schema_instance.entities["PERSON"]["properties"] == [
        {"description": "", "name": "birth date", "type": "ZONED_DATETIME"},
        {"description": "", "name": "name", "type": "STRING"},
    ]
    assert (
        schema_instance.entities["ORGANIZATION"]["description"]
        == "A structured group of people with a common purpose."
    )
    assert schema_instance.entities["AGE"]["description"] == "Age of a person in years."

    assert schema_instance.relations
    assert (
        schema_instance.relations["EMPLOYED_BY"]["description"]
        == "Indicates employment relationship."
    )
    assert (
        schema_instance.relations["ORGANIZED_BY"]["description"]
        == "Indicates organization responsible for an event."
    )
    assert (
        schema_instance.relations["ATTENDED_BY"]["description"]
        == "Indicates attendance at an event."
    )
    assert schema_instance.relations["EMPLOYED_BY"]["properties"] == [
        {"description": "", "name": "start_time", "type": "LOCAL_DATETIME"},
        {"description": "", "name": "end_time", "type": "LOCAL_DATETIME"},
    ]


def test_create_schema_model_no_relations_or_potential_schema(
    schema_builder: SchemaBuilder,
    valid_entities: list[SchemaEntity],
) -> None:
    schema_instance = schema_builder.create_schema_model(valid_entities)

    assert (
        schema_instance.entities["PERSON"]["description"]
        == "An individual human being."
    )
    assert schema_instance.entities["PERSON"]["properties"] == [
        {"description": "", "name": "birth date", "type": "ZONED_DATETIME"},
        {"description": "", "name": "name", "type": "STRING"},
    ]
    assert (
        schema_instance.entities["ORGANIZATION"]["description"]
        == "A structured group of people with a common purpose."
    )
    assert schema_instance.entities["AGE"]["description"] == "Age of a person in years."


def test_create_schema_model_missing_relations(
    schema_builder: SchemaBuilder,
    valid_entities: list[SchemaEntity],
    potential_schema: list[tuple[str, str, str]],
) -> None:
    with pytest.raises(SchemaValidationError) as exc_info:
        schema_builder.create_schema_model(
            entities=valid_entities, potential_schema=potential_schema
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
        "entities": [
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
        "relations": [
            {
                "label": "WORKS_FOR",
                "properties": [
                    {"name": "since", "type": "DATE"}
                ]
            }
        ],
        "potential_schema": [
            ["Person", "WORKS_FOR", "Organization"]
        ]
    }
    """


@pytest.fixture
def invalid_schema_json() -> str:
    return """
    {
        "entities": [
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
    assert len(schema_config.entities) == 2
    assert "Person" in schema_config.entities
    assert "Organization" in schema_config.entities

    assert schema_config.relations is not None
    assert "WORKS_FOR" in schema_config.relations

    assert schema_config.potential_schema is not None
    assert len(schema_config.potential_schema) == 1
    assert schema_config.potential_schema[0] == ("Person", "WORKS_FOR", "Organization")


@pytest.mark.asyncio
async def test_schema_from_text_run_invalid_json(
    schema_from_text: SchemaFromTextExtractor,
    mock_llm: AsyncMock,
    invalid_schema_json: str,
) -> None:
    # configure the mock LLM to return invalid JSON
    mock_llm.ainvoke.return_value = LLMResponse(content=invalid_schema_json)

    # verify that running with invalid JSON raises a ValueError
    with pytest.raises(ValueError) as exc_info:
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
async def test_schema_config_store_as_json(schema_config: SchemaConfig) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        # create file path
        json_path = os.path.join(temp_dir, "schema.json")

        # store the schema config
        schema_config.store_as_json(json_path)

        # verify the file exists and has content
        assert os.path.exists(json_path)
        assert os.path.getsize(json_path) > 0

        # verify the content is valid JSON and contains expected data
        with open(json_path, "r") as f:
            data = json.load(f)
            assert "entities" in data
            assert "PERSON" in data["entities"]
            assert "properties" in data["entities"]["PERSON"]
            assert "description" in data["entities"]["PERSON"]
            assert (
                data["entities"]["PERSON"]["description"]
                == "An individual human being."
            )


@pytest.mark.asyncio
async def test_schema_config_store_as_yaml(schema_config: SchemaConfig) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create file path
        yaml_path = os.path.join(temp_dir, "schema.yaml")

        # Store the schema config
        schema_config.store_as_yaml(yaml_path)

        # Verify the file exists and has content
        assert os.path.exists(yaml_path)
        assert os.path.getsize(yaml_path) > 0

        # Verify the content is valid YAML and contains expected data
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
            assert "entities" in data
            assert "PERSON" in data["entities"]
            assert "properties" in data["entities"]["PERSON"]
            assert "description" in data["entities"]["PERSON"]
            assert (
                data["entities"]["PERSON"]["description"]
                == "An individual human being."
            )


@pytest.mark.asyncio
async def test_schema_config_from_file(schema_config: SchemaConfig) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        # create file paths with different extensions
        json_path = os.path.join(temp_dir, "schema.json")
        yaml_path = os.path.join(temp_dir, "schema.yaml")
        yml_path = os.path.join(temp_dir, "schema.yml")

        # store the schema config in the different formats
        schema_config.store_as_json(json_path)
        schema_config.store_as_yaml(yaml_path)
        schema_config.store_as_yaml(yml_path)

        # load using from_file which should detect the format based on extension
        json_schema = SchemaConfig.from_file(json_path)
        yaml_schema = SchemaConfig.from_file(yaml_path)
        yml_schema = SchemaConfig.from_file(yml_path)

        # simple verification that the objects were loaded correctly
        assert isinstance(json_schema, SchemaConfig)
        assert isinstance(yaml_schema, SchemaConfig)
        assert isinstance(yml_schema, SchemaConfig)

        # verify basic structure is intact
        assert "entities" in json_schema.model_dump()
        assert "entities" in yaml_schema.model_dump()
        assert "entities" in yml_schema.model_dump()

        # verify an unsupported extension raises the correct error
        txt_path = os.path.join(temp_dir, "schema.txt")
        schema_config.store_as_json(txt_path)  # Store as JSON but with .txt extension

        with pytest.raises(ValueError, match="Unsupported file format"):
            SchemaConfig.from_file(txt_path)


@pytest.fixture
def valid_schema_json_array() -> str:
    return """
    [
        {
            "entities": [
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
            "relations": [
                {
                    "label": "WORKS_FOR",
                    "properties": [
                        {"name": "since", "type": "DATE"}
                    ]
                }
            ],
            "potential_schema": [
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
    schema_config = await schema_from_text.run(text="Sample text for extraction")

    # verify the schema was correctly extracted from the array
    assert len(schema_config.entities) == 2
    assert "Person" in schema_config.entities
    assert "Organization" in schema_config.entities

    assert schema_config.relations is not None
    assert "WORKS_FOR" in schema_config.relations

    assert schema_config.potential_schema is not None
    assert len(schema_config.potential_schema) == 1
    assert schema_config.potential_schema[0] == ("Person", "WORKS_FOR", "Organization")
