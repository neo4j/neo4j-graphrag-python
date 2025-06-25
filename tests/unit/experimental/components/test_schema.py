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
from unittest.mock import AsyncMock, patch

import neo4j
import pytest
from pydantic import ValidationError

from neo4j_graphrag.exceptions import SchemaValidationError, SchemaExtractionError, SchemaDatabaseConflictError
from neo4j_graphrag.experimental.components.schema import (
    SchemaBuilder,
    SchemaFromTextExtractor,
)
from neo4j_graphrag.experimental.components.types import (
    NodeType,
    PropertyType,
    RelationshipType,
    GraphSchema,
    Neo4jPropertyType,
    SchemaConstraint,
    Neo4jConstraintTypeEnum,
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


def test_node_type_raise_error_if_misconfigured() -> None:
    with pytest.raises(ValidationError):
        NodeType(
            label="test",
            properties=[],
            additional_properties=False,
        )


def test_relationship_type_initialization_from_string() -> None:
    relationship_type = RelationshipType.model_validate("REL")
    assert isinstance(relationship_type, RelationshipType)
    assert relationship_type.label == "REL"
    assert relationship_type.properties == []


def test_relationship_type_raise_error_if_misconfigured() -> None:
    with pytest.raises(ValidationError):
        RelationshipType(
            label="test",
            properties=[],
            additional_properties=False,
        )


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
        "additional_patterns": False,
    }
    with pytest.raises(
        ValidationError,
        match="`additional_relationship_types` must be set to False when using `additional_patterns=False`",
    ):
        GraphSchema.model_validate(schema_dict)


@pytest.fixture
def valid_node_types() -> tuple[NodeType, ...]:
    return (
        NodeType(
            label="PERSON",
            description="An individual human being.",
            properties=[
                PropertyType(name="birth date", type=Neo4jPropertyType.ZONED_DATE),
                PropertyType(name="name", type=Neo4jPropertyType.STRING, required=True),
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
                PropertyType(
                    name="start_time",
                    type=Neo4jPropertyType.LOCAL_DATETIME,
                    required=True,
                ),
                PropertyType(name="end_time", type=Neo4jPropertyType.LOCAL_DATETIME),
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
def patterns_with_invalid_relation() -> tuple[tuple[str, str, str], ...]:
    return (("PERSON", "NON_EXISTENT_RELATION", "ORGANIZATION"),)


@pytest.fixture
def schema_builder(driver: neo4j.Driver) -> SchemaBuilder:
    return SchemaBuilder(driver)


@pytest.fixture
def graph_schema(
    schema_builder: SchemaBuilder,
    valid_node_types: Tuple[NodeType, ...],
    valid_relationship_types: Tuple[RelationshipType, ...],
    valid_patterns: Tuple[Tuple[str, str, str], ...],
) -> GraphSchema:
    return schema_builder._create_schema_model(
        list(valid_node_types), list(valid_relationship_types), list(valid_patterns)
    )


def test_create_schema_model_valid_data(
    schema_builder: SchemaBuilder,
    valid_node_types: Tuple[NodeType, ...],
    valid_relationship_types: Tuple[RelationshipType, ...],
    valid_patterns: Tuple[Tuple[str, str, str], ...],
) -> None:
    schema = schema_builder._create_schema_model(
        list(valid_node_types), list(valid_relationship_types), list(valid_patterns)
    )

    assert schema.node_types == valid_node_types
    assert schema.relationship_types == valid_relationship_types
    assert schema.patterns == valid_patterns
    assert schema.additional_node_types is True
    assert schema.additional_relationship_types is True
    assert schema.additional_patterns is True


@pytest.mark.asyncio
async def test_run_method(
    schema_builder: SchemaBuilder,
    valid_node_types: Tuple[NodeType, ...],
    valid_relationship_types: Tuple[RelationshipType, ...],
    valid_patterns: Tuple[Tuple[str, str, str], ...],
) -> None:
    with patch.object(
        schema_builder,
        "_create_schema_model",
        return_value=GraphSchema(
            node_types=valid_node_types,
            relationship_types=valid_relationship_types,
            patterns=valid_patterns,
        ),
    ):
        # Call with strings instead of NodeType objects
        schema = await schema_builder.run(
            node_types=["PERSON", "ORGANIZATION", "AGE"],
            relationship_types=["EMPLOYED_BY", "ORGANIZED_BY", "ATTENDED_BY"],
            patterns=valid_patterns
        )

    assert schema.node_types == valid_node_types
    assert schema.relationship_types == valid_relationship_types
    assert schema.patterns == valid_patterns
    assert schema.additional_node_types is True
    assert schema.additional_relationship_types is True
    assert schema.additional_patterns is True


def test_create_schema_model_invalid_entity(
    schema_builder: SchemaBuilder,
    valid_node_types: Tuple[NodeType, ...],
    valid_relationship_types: Tuple[RelationshipType, ...],
    patterns_with_invalid_entity: Tuple[Tuple[str, str, str], ...],
) -> None:
    with pytest.raises(SchemaValidationError) as exc_info:
        schema_builder._create_schema_model(
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
        schema_builder._create_schema_model(
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
    schema_instance = schema_builder._create_schema_model(
        list(valid_node_types), list(valid_relationship_types)
    )
    assert schema_instance.node_types == valid_node_types
    assert schema_instance.relationship_types == valid_relationship_types
    assert schema_instance.patterns == ()


def test_create_schema_model_no_relations_or_potential_schema(
    schema_builder: SchemaBuilder,
    valid_node_types: Tuple[NodeType, ...],
) -> None:
    schema_instance = schema_builder._create_schema_model(list(valid_node_types))

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
        schema_builder._create_schema_model(
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
def schema_from_text(
    driver: neo4j.Driver, mock_llm: AsyncMock
) -> SchemaFromTextExtractor:
    return SchemaFromTextExtractor(driver, llm=mock_llm)


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
        driver=None, llm=mock_llm, prompt_template=custom_template
    )

    # configure mock LLM to return valid JSON and capture the prompt that was sent to it
    mock_llm.ainvoke.return_value = LLMResponse(content=valid_schema_json)

    # Mock constraint retrieval to avoid database access
    with patch.object(schema_from_text, '_get_constraints_from_db', return_value=[]):
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
    schema_from_text = SchemaFromTextExtractor(driver=None, llm=mock_llm, llm_params=llm_params)

    # configure the mock LLM to return a valid schema JSON
    mock_llm.ainvoke.return_value = LLMResponse(content=valid_schema_json)

    # Mock constraint retrieval to avoid database access
    with patch.object(schema_from_text, '_get_constraints_from_db', return_value=[]):
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


# ==================== CONFLICT DETECTION TESTS ====================

@pytest.fixture
def mock_constraints_missing_property() -> list[SchemaConstraint]:
    """Mock constraints that reference properties not in user schema."""
    return [
        SchemaConstraint(
            entity_type="NODE",
            label_or_type=["PERSON"],
            type=Neo4jConstraintTypeEnum.NODE_PROPERTY_EXISTENCE,
            properties=["missing_property"],
        )
    ]


@pytest.fixture
def mock_constraints_type_conflict() -> list[SchemaConstraint]:
    """Mock constraints with type conflicts."""
    return [
        SchemaConstraint(
            entity_type="NODE",
            label_or_type=["PERSON"],
            type=Neo4jConstraintTypeEnum.NODE_PROPERTY_TYPE,
            properties=["name"],
            property_type=[Neo4jPropertyType.INTEGER],  # Conflicts with STRING
        )
    ]


@pytest.fixture
def mock_constraints_required_conflict() -> list[SchemaConstraint]:
    """Mock constraints requiring properties marked as optional by user."""
    return [
        SchemaConstraint(
            entity_type="NODE",
            label_or_type=["PERSON"],
            type=Neo4jConstraintTypeEnum.NODE_PROPERTY_EXISTENCE,
            properties=["optional_prop"],
        )
    ]


@pytest.fixture
def mock_constraints_missing_entity() -> list[SchemaConstraint]:
    """Mock constraints on entity types not in user schema."""
    return [
        SchemaConstraint(
            entity_type="NODE",
            label_or_type=["UNKNOWN_LABEL"],
            type=Neo4jConstraintTypeEnum.NODE_PROPERTY_EXISTENCE,
            properties=["some_property"],
        )
    ]


@pytest.fixture
def person_node_with_optional_prop() -> NodeType:
    """Person node with an optional property that conflicts with DB requirements."""
    return NodeType(
        label="PERSON",
        properties=[
            PropertyType(name="name", type=Neo4jPropertyType.STRING, required=True),
            PropertyType(name="optional_prop", type=Neo4jPropertyType.STRING, required=False),
        ],
    )


@pytest.fixture
def person_node_additional_props_false() -> NodeType:
    """Person node with additional_properties=False."""
    return NodeType(
        label="PERSON",
        properties=[
            PropertyType(name="name", type=Neo4jPropertyType.STRING, required=True),
        ],
        additional_properties=False,
    )


def test_missing_property_conflict(
    schema_builder: SchemaBuilder, mock_constraints_missing_property: list[SchemaConstraint]
) -> None:
    """Test that missing properties in user schema raise appropriate error."""
    with patch.object(schema_builder, '_get_constraints_from_db', return_value=mock_constraints_missing_property):
        with pytest.raises(SchemaDatabaseConflictError, match="requires properties \\['missing_property'\\]"):
            schema_builder._create_schema_model([
                NodeType(label="PERSON", properties=[
                    PropertyType(name="name", type=Neo4jPropertyType.STRING)
                ])
            ])


def test_property_type_conflict(
    schema_builder: SchemaBuilder, mock_constraints_type_conflict: list[SchemaConstraint]
) -> None:
    """Test that property type conflicts raise appropriate error."""
    with patch.object(schema_builder, '_get_constraints_from_db', return_value=mock_constraints_type_conflict):
        with pytest.raises(SchemaDatabaseConflictError, match="has type .* but database constraint allows only"):
            schema_builder._create_schema_model([
                NodeType(label="PERSON", properties=[
                    PropertyType(name="name", type=Neo4jPropertyType.STRING)  # Conflicts with INTEGER
                ])
            ])


def test_required_property_conflict(
    schema_builder: SchemaBuilder, 
    mock_constraints_required_conflict: list[SchemaConstraint],
    person_node_with_optional_prop: NodeType
) -> None:
    """Test that optional properties conflicting with DB existence constraints are enhanced, not errors."""
    with patch.object(schema_builder, '_get_constraints_from_db', return_value=mock_constraints_required_conflict):
        # This should not raise an exception - we enhance instead of error
        result = schema_builder._create_schema_model([person_node_with_optional_prop])
        
        # Property should be enhanced to required=True
        optional_prop = None
        for prop in result.node_types[0].properties:
            if prop.name == "optional_prop":
                optional_prop = prop
                break
        
        assert optional_prop is not None
        assert optional_prop.required is True  # Should be enhanced


def test_missing_entity_type_conflict(
    schema_builder: SchemaBuilder, mock_constraints_missing_entity: list[SchemaConstraint]
) -> None:
    """Test that missing entity types raise error when additional types are disabled."""
    with patch.object(schema_builder, '_get_constraints_from_db', return_value=mock_constraints_missing_entity):
        with pytest.raises(SchemaDatabaseConflictError, match="Database has constraints on node labels"):
            schema_builder._create_schema_model(
                [NodeType(label="PERSON")],
                additional_node_types=False
            )


def test_additional_properties_conflict(
    schema_builder: SchemaBuilder, 
    person_node_additional_props_false: NodeType
) -> None:
    """Test that additional_properties=False conflicts with DB-required properties."""
    mock_constraints = [
        SchemaConstraint(
            entity_type="NODE",
            label_or_type=["PERSON"],
            type=Neo4jConstraintTypeEnum.NODE_PROPERTY_EXISTENCE,
            properties=["db_required_prop"],
        )
    ]
    
    with patch.object(schema_builder, '_get_constraints_from_db', return_value=mock_constraints):
        with pytest.raises(SchemaDatabaseConflictError, match="has additional_properties=False but database.*require"):
            schema_builder._create_schema_model([person_node_additional_props_false])


def test_no_conflict_with_compatible_schema(
    schema_builder: SchemaBuilder
) -> None:
    """Test that compatible schema and constraints work without errors."""
    mock_constraints = [
        SchemaConstraint(
            entity_type="NODE",
            label_or_type=["PERSON"],
            type=Neo4jConstraintTypeEnum.NODE_PROPERTY_EXISTENCE,
            properties=["name"],
        )
    ]
    
    with patch.object(schema_builder, '_get_constraints_from_db', return_value=mock_constraints):
        # This should not raise any exceptions
        result = schema_builder._create_schema_model([
            NodeType(label="PERSON", properties=[
                PropertyType(name="name", type=Neo4jPropertyType.STRING, required=True)
            ])
        ])
        
        assert len(result.node_types) == 1
        assert result.node_types[0].label == "PERSON"
        # Property should remain required=True
        assert result.node_types[0].properties[0].required is True


def test_enhancement_sets_required_property(
    schema_builder: SchemaBuilder
) -> None:
    """Test that existence constraints properly set required=True on properties."""
    mock_constraints = [
        SchemaConstraint(
            entity_type="NODE",
            label_or_type=["PERSON"],
            type=Neo4jConstraintTypeEnum.NODE_PROPERTY_EXISTENCE,
            properties=["name"],
        )
    ]
    
    with patch.object(schema_builder, '_get_constraints_from_db', return_value=mock_constraints):
        result = schema_builder._create_schema_model([
            NodeType(label="PERSON", properties=[
                PropertyType(name="name", type=Neo4jPropertyType.STRING, required=False)
            ])
        ])
        
        # Property should be enhanced to required=True
        assert result.node_types[0].properties[0].required is True


def test_compatible_property_types(
    schema_builder: SchemaBuilder
) -> None:
    """Test that compatible property types don't raise conflicts."""
    mock_constraints = [
        SchemaConstraint(
            entity_type="NODE",
            label_or_type=["PERSON"],
            type=Neo4jConstraintTypeEnum.NODE_PROPERTY_TYPE,
            properties=["name"],
            property_type=[Neo4jPropertyType.STRING, Neo4jPropertyType.INTEGER],  # Union type
        )
    ]
    
    with patch.object(schema_builder, '_get_constraints_from_db', return_value=mock_constraints):
        # User specifies STRING, DB allows STRING|INTEGER - should be compatible
        result = schema_builder._create_schema_model([
            NodeType(label="PERSON", properties=[
                PropertyType(name="name", type=Neo4jPropertyType.STRING)
            ])
        ])
        
        assert len(result.node_types) == 1
        assert result.node_types[0].properties[0].type == Neo4jPropertyType.STRING


def test_missing_entity_allowed_with_additional_types(
    schema_builder: SchemaBuilder, mock_constraints_missing_entity: list[SchemaConstraint]
) -> None:
    """Test that missing entity types are allowed when additional_*_types=True."""
    with patch.object(schema_builder, '_get_constraints_from_db', return_value=mock_constraints_missing_entity):
        # This should not raise an exception because additional_node_types=True by default
        result = schema_builder._create_schema_model([
            NodeType(label="PERSON")
        ])
        
        assert len(result.node_types) == 1
        assert result.node_types[0].label == "PERSON"


def test_relationship_constraint_conflicts(
    schema_builder: SchemaBuilder
) -> None:
    """Test conflict detection for relationship constraints."""
    mock_constraints = [
        SchemaConstraint(
            entity_type="RELATIONSHIP",
            label_or_type=["KNOWS"],
            type=Neo4jConstraintTypeEnum.RELATIONSHIP_PROPERTY_EXISTENCE,
            properties=["missing_rel_prop"],
        )
    ]
    
    with patch.object(schema_builder, '_get_constraints_from_db', return_value=mock_constraints):
        with pytest.raises(SchemaDatabaseConflictError, match="requires properties \\['missing_rel_prop'\\]"):
            schema_builder._create_schema_model(
                [NodeType(label="PERSON")],
                [RelationshipType(label="KNOWS", properties=[
                    PropertyType(name="since", type=Neo4jPropertyType.DATE)
                ])]
            )


# ==================== SCHEMA FROM TEXT EXTRACTOR ENHANCEMENT TESTS ====================

@pytest.fixture
def schema_from_text_extractor(driver: neo4j.Driver) -> SchemaFromTextExtractor:
    """Fixture providing a SchemaFromTextExtractor instance."""
    from neo4j_graphrag.llm.base import LLMInterface
    from neo4j_graphrag.llm.types import LLMResponse
    
    class MockLLM(LLMInterface):
        def __init__(self, response_content: str):
            super().__init__(model_name="mock-model")
            self.response_content = response_content
            
        async def ainvoke(self, input_: str, **kwargs: Any) -> LLMResponse:
            return LLMResponse(content=self.response_content)
            
        def invoke(self, input_: str, **kwargs: Any) -> LLMResponse:
            return LLMResponse(content=self.response_content)
    
    # Mock LLM that returns a basic schema
    mock_llm = MockLLM('{"node_types": [{"label": "Person", "properties": [{"name": "name", "type": "STRING"}]}], "relationship_types": [], "patterns": []}')
    
    return SchemaFromTextExtractor(
        driver=driver,
        llm=mock_llm
    )


def test_schema_enhancement_adds_missing_properties(
    schema_from_text_extractor: SchemaFromTextExtractor,
    mock_constraints_missing_property: list[SchemaConstraint]
) -> None:
    """Test that enhancement adds missing properties required by constraints."""
    # Create a basic schema missing the required property
    initial_schema = GraphSchema(
        node_types=[
            NodeType(label="PERSON", properties=[
                PropertyType(name="name", type="STRING")
            ])
        ],
        relationship_types=[],
        patterns=[]
    )
    
    with patch.object(schema_from_text_extractor, '_get_constraints_from_db', return_value=mock_constraints_missing_property):
        enhanced_schema = schema_from_text_extractor._process_constraints_against_schema(initial_schema, mode="enhance")
        
        # Check that the missing property was added
        person_node = enhanced_schema.node_type_from_label("PERSON")
        missing_prop = person_node.get_property_by_name("missing_property")
        
        assert missing_prop is not None
        assert missing_prop.required == True
        assert "constraint" in missing_prop.description.lower()


def test_schema_enhancement_adds_missing_entity_types(
    schema_from_text_extractor: SchemaFromTextExtractor
) -> None:
    """Test that enhancement adds missing entity types required by constraints."""
    # Mock constraints requiring an entity type not in the schema
    mock_constraints = [
        SchemaConstraint(
            entity_type="NODE",
            label_or_type=["ORGANIZATION"],
            type=Neo4jConstraintTypeEnum.NODE_PROPERTY_EXISTENCE,
            properties=["name"],
        )
    ]
    
    # Schema without ORGANIZATION
    initial_schema = GraphSchema(
        node_types=[
            NodeType(label="PERSON", properties=[
                PropertyType(name="name", type="STRING")
            ])
        ],
        relationship_types=[],
        patterns=[]
    )
    
    with patch.object(schema_from_text_extractor, '_get_constraints_from_db', return_value=mock_constraints):
        enhanced_schema = schema_from_text_extractor._process_constraints_against_schema(initial_schema, mode="enhance")
        
        # Check that ORGANIZATION was added
        org_node = enhanced_schema.node_type_from_label("ORGANIZATION")
        assert org_node is not None
        assert "constraint" in org_node.description.lower()
        
        # Check that required property was added
        name_prop = org_node.get_property_by_name("name")
        assert name_prop is not None
        assert name_prop.required == True


def test_schema_enhancement_updates_property_types(
    schema_from_text_extractor: SchemaFromTextExtractor
) -> None:
    """Test that enhancement updates property types to match constraints."""
    # Mock type constraint
    mock_constraints = [
        SchemaConstraint(
            entity_type="NODE",
            label_or_type=["PERSON"],
            type=Neo4jConstraintTypeEnum.NODE_PROPERTY_TYPE,
            properties=["age"],
            property_type=[Neo4jPropertyType.INTEGER]
        )
    ]
    
    # Schema with wrong type
    initial_schema = GraphSchema(
        node_types=[
            NodeType(label="PERSON", properties=[
                PropertyType(name="age", type="STRING")  # Wrong type
            ])
        ],
        relationship_types=[],
        patterns=[]
    )
    
    with patch.object(schema_from_text_extractor, '_get_constraints_from_db', return_value=mock_constraints):
        enhanced_schema = schema_from_text_extractor._process_constraints_against_schema(initial_schema, mode="enhance")
        
        # Check that property type was updated
        person_node = enhanced_schema.node_type_from_label("PERSON")
        age_prop = person_node.get_property_by_name("age")
        assert age_prop.type == Neo4jPropertyType.INTEGER


def test_schema_enhancement_sets_required_properties(
    schema_from_text_extractor: SchemaFromTextExtractor
) -> None:
    """Test that enhancement sets required=True for existence constraints."""
    mock_constraints = [
        SchemaConstraint(
            entity_type="NODE",
            label_or_type=["PERSON"],
            type=Neo4jConstraintTypeEnum.NODE_PROPERTY_EXISTENCE,
            properties=["email"],
        )
    ]
    
    # Schema with optional property
    initial_schema = GraphSchema(
        node_types=[
            NodeType(label="PERSON", properties=[
                PropertyType(name="email", type="STRING", required=False)
            ])
        ],
        relationship_types=[],
        patterns=[]
    )
    
    with patch.object(schema_from_text_extractor, '_get_constraints_from_db', return_value=mock_constraints):
        enhanced_schema = schema_from_text_extractor._process_constraints_against_schema(initial_schema, mode="enhance")
        
        # Check that property was made required
        person_node = enhanced_schema.node_type_from_label("PERSON")
        email_prop = person_node.get_property_by_name("email")
        assert email_prop.required == True


def test_schema_enhancement_handles_relationship_constraints(
    schema_from_text_extractor: SchemaFromTextExtractor
) -> None:
    """Test that enhancement works for relationship constraints."""
    mock_constraints = [
        SchemaConstraint(
            entity_type="RELATIONSHIP",
            label_or_type=["KNOWS"],
            type=Neo4jConstraintTypeEnum.RELATIONSHIP_PROPERTY_EXISTENCE,
            properties=["since"],
        )
    ]
    
    # Schema without the relationship property
    initial_schema = GraphSchema(
        node_types=[],
        relationship_types=[
            RelationshipType(label="KNOWS", properties=[])
        ],
        patterns=[]
    )
    
    with patch.object(schema_from_text_extractor, '_get_constraints_from_db', return_value=mock_constraints):
        enhanced_schema = schema_from_text_extractor._process_constraints_against_schema(initial_schema, mode="enhance")
        
        # Check that property was added to relationship
        knows_rel = enhanced_schema.relationship_type_from_label("KNOWS")
        since_prop = knows_rel.get_property_by_name("since")
        assert since_prop is not None
        assert since_prop.required == True


def test_schema_enhancement_respects_additional_properties_false(
    schema_from_text_extractor: SchemaFromTextExtractor
) -> None:
    """Test that enhancement respects additional_properties=False."""
    mock_constraints = [
        SchemaConstraint(
            entity_type="NODE",
            label_or_type=["PERSON"],
            type=Neo4jConstraintTypeEnum.NODE_PROPERTY_EXISTENCE,
            properties=["missing_prop"],
        )
    ]
    
    # Schema with additional_properties=False
    initial_schema = GraphSchema(
        node_types=[
            NodeType(
                label="PERSON", 
                properties=[PropertyType(name="name", type="STRING")],
                additional_properties=False
            )
        ],
        relationship_types=[],
        patterns=[]
    )
    
    with patch.object(schema_from_text_extractor, '_get_constraints_from_db', return_value=mock_constraints):
        enhanced_schema = schema_from_text_extractor._process_constraints_against_schema(initial_schema, mode="enhance")
        
        # Check that property was NOT added due to additional_properties=False
        person_node = enhanced_schema.node_type_from_label("PERSON")
        missing_prop = person_node.get_property_by_name("missing_prop")
        assert missing_prop is None


def test_schema_enhancement_handles_no_constraints(
    schema_from_text_extractor: SchemaFromTextExtractor
) -> None:
    """Test that enhancement returns original schema when no constraints exist."""
    initial_schema = GraphSchema(
        node_types=[
            NodeType(label="PERSON", properties=[
                PropertyType(name="name", type="STRING")
            ])
        ],
        relationship_types=[],
        patterns=[]
    )
    
    with patch.object(schema_from_text_extractor, '_get_constraints_from_db', return_value=[]):
        enhanced_schema = schema_from_text_extractor._process_constraints_against_schema(initial_schema, mode="enhance")
        
        # Should return the same schema
        assert enhanced_schema.model_dump() == initial_schema.model_dump()


@pytest.mark.asyncio
async def test_schema_from_text_extractor_run_with_enhancement(
    driver: neo4j.Driver
) -> None:
    """Test that the run method applies enhancement to LLM-generated schema."""
    from neo4j_graphrag.llm.base import LLMInterface
    from neo4j_graphrag.llm.types import LLMResponse
    
    class MockLLM(LLMInterface):
        def __init__(self):
            super().__init__(model_name="mock-model")
            
        async def ainvoke(self, input_: str, **kwargs: Any) -> LLMResponse:
            # Return a schema missing properties that will be required by constraints
            return LLMResponse(content='{"node_types": [{"label": "PERSON", "properties": [{"name": "name", "type": "STRING"}]}], "relationship_types": [], "patterns": []}')
            
        def invoke(self, input_: str, **kwargs: Any) -> LLMResponse:
            return LLMResponse(content='{"node_types": [{"label": "PERSON", "properties": [{"name": "name", "type": "STRING"}]}], "relationship_types": [], "patterns": []}')
    
    # Mock constraints that require additional properties
    mock_constraints = [
        SchemaConstraint(
            entity_type="NODE",
            label_or_type=["PERSON"],
            type=Neo4jConstraintTypeEnum.NODE_PROPERTY_EXISTENCE,
            properties=["email"],
        )
    ]
    
    extractor = SchemaFromTextExtractor(
        driver=driver,
        llm=MockLLM()
    )
    
    with patch.object(extractor, '_get_constraints_from_db', return_value=mock_constraints):
        enhanced_schema = await extractor.run("Some text about persons")
        
        # Check that the schema was enhanced with the missing property
        person_node = enhanced_schema.node_type_from_label("PERSON")
        email_prop = person_node.get_property_by_name("email")
        assert email_prop is not None
        assert email_prop.required == True
        assert "constraint" in email_prop.description.lower()


def test_schema_enhancement_graceful_failure(
    schema_from_text_extractor: SchemaFromTextExtractor
) -> None:
    """Test that enhancement fails gracefully and returns original schema."""
    initial_schema = GraphSchema(
        node_types=[
            NodeType(label="PERSON", properties=[
                PropertyType(name="name", type="STRING")
            ])
        ],
        relationship_types=[],
        patterns=[]
    )
    
    # Mock constraints that will cause validation error
    mock_constraints = [
        SchemaConstraint(
            entity_type="NODE",
            label_or_type=["PERSON"],
            type=Neo4jConstraintTypeEnum.NODE_PROPERTY_EXISTENCE,
            properties=["problematic_prop"],
        )
    ]
    
    with patch.object(schema_from_text_extractor, '_get_constraints_from_db', return_value=mock_constraints):
        with patch('neo4j_graphrag.experimental.components.schema.GraphSchema.model_validate', side_effect=ValidationError.from_exception_data("GraphSchema", [])):
            # Should return original schema on validation failure
            result = schema_from_text_extractor._process_constraints_against_schema(initial_schema, mode="enhance")
            assert result.model_dump() == initial_schema.model_dump()


def test_schema_builder_enhancement_mode_flag(driver: neo4j.Driver) -> None:
    """Test that SchemaBuilder enhancement_mode flag switches between validation and enhancement."""
    # Test validation mode (default)
    validation_builder = SchemaBuilder(driver, enhancement_mode=False)
    
    mock_constraints = [
        SchemaConstraint(
            entity_type="NODE",
            label_or_type=["PERSON"],
            type=Neo4jConstraintTypeEnum.NODE_PROPERTY_EXISTENCE,
            properties=["missing_property"],
        )
    ]
    
    with patch.object(validation_builder, '_get_constraints_from_db', return_value=mock_constraints):
        # Should raise error in validation mode
        with pytest.raises(SchemaDatabaseConflictError, match="requires properties \\['missing_property'\\]"):
            validation_builder._create_schema_model([
                NodeType(label="PERSON", properties=[
                    PropertyType(name="name", type="STRING")
                ])
            ])
    
    # Test enhancement mode
    enhancement_builder = SchemaBuilder(driver, enhancement_mode=True)
    
    with patch.object(enhancement_builder, '_get_constraints_from_db', return_value=mock_constraints):
        # Should enhance schema instead of raising error
        result = enhancement_builder._create_schema_model([
            NodeType(label="PERSON", properties=[
                PropertyType(name="name", type="STRING")
            ])
        ])
        
        # Check that missing property was added
        person_node = result.node_type_from_label("PERSON")
        missing_prop = person_node.get_property_by_name("missing_property")
        assert missing_prop is not None
        assert missing_prop.required == True
        assert "constraint" in missing_prop.description.lower()
