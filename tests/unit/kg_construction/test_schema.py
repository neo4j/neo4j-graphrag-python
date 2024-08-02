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
from neo4j_genai.kg_construction.schema import Entity, Relation, SchemaBuilder
from pydantic import ValidationError


@pytest.fixture
def valid_entities() -> list[Entity]:
    return [
        Entity(
            name="PERSON",
            type="str",
            description="An individual human being.",
            properties=["birth date", "name"],
        ),
        Entity(
            name="ORGANIZATION",
            type="str",
            description="A structured group of people with a common purpose.",
        ),
        Entity(name="AGE", type="int", description="Age of a person in years."),
    ]


@pytest.fixture
def valid_relations() -> list[Relation]:
    return [
        Relation(
            name="EMPLOYED_BY",
            description="Indicates employment relationship.",
            properties=["start_time", "end_time"],
        ),
        Relation(
            name="ORGANIZED_BY",
            description="Indicates organization responsible for an event.",
        ),
        Relation(name="ATTENDED_BY", description="Indicates attendance at an event."),
    ]


@pytest.fixture
def potential_schema() -> dict[str, list[str]]:
    return {
        "PERSON": ["EMPLOYED_BY", "ATTENDED_BY"],
        "ORGANIZATION": ["EMPLOYED_BY", "ORGANIZED_BY"],
    }


@pytest.fixture
def potential_schema_with_invalid_entity() -> dict[str, list[str]]:
    return {
        "PERSON": ["EMPLOYED_BY"],
        "NON_EXISTENT_ENTITY": ["ORGANIZED_BY"],
    }


@pytest.fixture
def potential_schema_with_invalid_relation() -> dict[str, list[str]]:
    return {
        "PERSON": ["EMPLOYED_BY", "NON_EXISTENT_RELATION"],
    }


@pytest.fixture
def schema_builder() -> SchemaBuilder:
    return SchemaBuilder()


def test_create_schema_model_valid_data(
    schema_builder: SchemaBuilder,
    valid_entities: list[Entity],
    valid_relations: list[Relation],
    potential_schema: dict[str, list[str]],
) -> None:
    schema_instance = schema_builder.create_schema_model(
        valid_entities, valid_relations, potential_schema
    )

    assert (
        schema_instance.entities["PERSON"]["description"]
        == "An individual human being."
    )
    assert schema_instance.entities["PERSON"]["properties"] == ["birth date", "name"]
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
        "start_time",
        "end_time",
    ]

    assert schema_instance.potential_schema == potential_schema


def test_create_schema_model_missing_description(schema_builder: SchemaBuilder) -> None:
    entities = [
        Entity(name="PERSON", type="str", description="An individual human being."),
        Entity(name="ORGANIZATION", type="str", description=""),
        Entity(name="AGE", type="int", description=""),
    ]
    relations = [
        Relation(name="EMPLOYED_BY", description="Indicates employment relationship."),
        Relation(name="ORGANIZED_BY", description=""),
        Relation(name="ATTENDED_BY", description=""),
    ]
    potential_schema = {
        "PERSON": ["EMPLOYED_BY", "ATTENDED_BY"],
        "ORGANIZATION": ["EMPLOYED_BY", "ORGANIZED_BY"],
    }

    schema_instance = schema_builder.create_schema_model(
        entities, relations, potential_schema
    )

    assert schema_instance.entities["ORGANIZATION"]["description"] == ""
    assert schema_instance.entities["AGE"]["description"] == ""
    assert schema_instance.relations["ORGANIZED_BY"]["description"] == ""
    assert schema_instance.relations["ATTENDED_BY"]["description"] == ""


def test_create_schema_model_empty_lists(schema_builder: SchemaBuilder) -> None:
    schema_instance = schema_builder.create_schema_model([], [], {})

    assert schema_instance.entities == {}
    assert schema_instance.relations == {}
    assert schema_instance.potential_schema == {}


def test_create_schema_model_invalid_data_types(schema_builder: SchemaBuilder) -> None:
    with pytest.raises(ValidationError):
        entities = [
            Entity(name="PERSON", type="str", description="An individual human being."),
            Entity(
                name="ORGANIZATION",
                type=123,  # type: ignore
                description="A structured group of people with a common purpose.",
            ),
        ]
        relations = [
            Relation(
                name="EMPLOYED_BY", description="Indicates employment relationship."
            ),
            Relation(
                name=456,  # type: ignore
                description="Indicates organization responsible for an event.",
            ),
        ]
        potential_schema = {
            "PERSON": ["EMPLOYED_BY", "ATTENDED_BY"],
            "ORGANIZATION": ["EMPLOYED_BY", "ORGANIZED_BY"],
        }

        schema_builder.create_schema_model(entities, relations, potential_schema)


def test_create_schema_model_invalid_properties_types(
    schema_builder: SchemaBuilder,
) -> None:
    with pytest.raises(ValidationError):
        entities = [
            Entity(
                name="PERSON",
                type="str",
                description="An individual human being.",
                properties=[42, 1337],
            ),
            Entity(
                name="ORGANIZATION",
                type="str",
                description="A structured group of people with a common purpose.",
            ),
        ]
        relations = [
            Relation(
                name="EMPLOYED_BY",
                description="Indicates employment relationship.",
                properties=[42, 1337],
            ),
            Relation(
                name="ORGANIZED_BY",
                description="Indicates organization responsible for an event.",
            ),
        ]
        potential_schema = {
            "PERSON": ["EMPLOYED_BY", "ATTENDED_BY"],
            "ORGANIZATION": ["EMPLOYED_BY", "ORGANIZED_BY"],
        }

        schema_builder.create_schema_model(entities, relations, potential_schema)


@pytest.mark.asyncio
async def test_run_method(
    schema_builder: SchemaBuilder,
    valid_entities: list[Entity],
    valid_relations: list[Relation],
    potential_schema: dict[str, list[str]],
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
    valid_entities: list[Entity],
    valid_relations: list[Relation],
    potential_schema_with_invalid_entity: dict[str, list[str]],
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
    valid_entities: list[Entity],
    valid_relations: list[Relation],
    potential_schema_with_invalid_relation: dict[str, list[str]],
) -> None:
    with pytest.raises(SchemaValidationError) as exc_info:
        schema_builder.create_schema_model(
            valid_entities, valid_relations, potential_schema_with_invalid_relation
        )
    assert "Relation 'NON_EXISTENT_RELATION' is not defined" in str(
        exc_info.value
    ), "Should fail due to non-existent relation"


def test_create_schema_model_missing_properties(schema_builder: SchemaBuilder) -> None:
    entities = [
        Entity(name="PERSON", type="str", description="An individual human being."),
        Entity(
            name="ORGANIZATION",
            type="str",
            description="A structured group of people with a common purpose.",
        ),
        Entity(name="AGE", type="int", description="Age of a person in years."),
    ]

    relations = [
        Relation(name="EMPLOYED_BY", description="Indicates employment relationship."),
        Relation(
            name="ORGANIZED_BY",
            description="Indicates organization responsible for an event.",
        ),
        Relation(name="ATTENDED_BY", description="Indicates attendance at an event."),
    ]

    schema_instance = schema_builder.create_schema_model(
        entities, relations, schema_builder
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
