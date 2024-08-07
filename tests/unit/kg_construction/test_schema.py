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

import pytest
from neo4j_genai.exceptions import SchemaValidationError
from neo4j_genai.kg_construction.schema import (
    SchemaBuilder,
    SchemaEntity,
    SchemaProperty,
    SchemaRelation,
)
from pydantic import ValidationError


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

    assert (
        schema_instance.relations["EMPLOYED_BY"]["properties"] == []
    ), "Expected empty properties for EMPLOYED_BY"
    assert (
        schema_instance.relations["ORGANIZED_BY"]["properties"] == []
    ), "Expected empty properties for ORGANIZED_BY"
    assert (
        schema_instance.relations["ATTENDED_BY"]["properties"] == []
    ), "Expected empty properties for ATTENDED_BY"
