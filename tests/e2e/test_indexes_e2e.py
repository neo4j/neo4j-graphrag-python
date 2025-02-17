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
import neo4j
import pytest
from neo4j_graphrag.indexes import (
    retrieve_fulltext_index_info,
    retrieve_vector_index_info,
    upsert_vectors,
)
from neo4j_graphrag.types import EntityType


@pytest.mark.usefixtures("setup_neo4j_for_retrieval")
def test_retrieve_vector_index_info_happy_path(driver: neo4j.Driver) -> None:
    index_info = retrieve_vector_index_info(
        driver=driver,
        index_name="vector-index-name",
        label_or_type="Document",
        embedding_property="vectorProperty",
    )
    assert index_info is not None
    index_name = index_info.get("name")
    assert index_name == "vector-index-name"
    index_type = index_info.get("type")
    assert index_type == "VECTOR"
    labels_or_types = index_info.get("labelsOrTypes")
    assert labels_or_types == ["Document"]
    properties = index_info.get("properties")
    assert properties == ["vectorProperty"]
    entity_type = index_info.get("entityType")
    assert entity_type == "NODE"
    options = index_info.get("options")
    assert isinstance(options, dict)
    index_config = options.get("indexConfig")
    assert isinstance(index_config, dict)
    embedding_dimension = index_config.get("vector.dimensions")
    assert isinstance(embedding_dimension, int)
    assert embedding_dimension == 1536


@pytest.mark.usefixtures("setup_neo4j_for_retrieval")
def test_retrieve_vector_index_info_no_index_name(driver: neo4j.Driver) -> None:
    index_info = retrieve_vector_index_info(
        driver=driver,
        index_name="",
        label_or_type="Document",
        embedding_property="vectorProperty",
    )
    assert index_info is not None
    index_name = index_info.get("name")
    assert index_name == "vector-index-name"
    index_type = index_info.get("type")
    assert index_type == "VECTOR"
    labels_or_types = index_info.get("labelsOrTypes")
    assert labels_or_types == ["Document"]
    properties = index_info.get("properties")
    assert properties == ["vectorProperty"]
    entity_type = index_info.get("entityType")
    assert entity_type == "NODE"
    options = index_info.get("options")
    assert isinstance(options, dict)
    index_config = options.get("indexConfig")
    assert isinstance(index_config, dict)
    embedding_dimension = index_config.get("vector.dimensions")
    assert isinstance(embedding_dimension, int)
    assert embedding_dimension == 1536


@pytest.mark.usefixtures("setup_neo4j_for_retrieval")
def test_retrieve_vector_index_info_no_label_or_property(driver: neo4j.Driver) -> None:
    index_info = retrieve_vector_index_info(
        driver=driver,
        index_name="vector-index-name",
        label_or_type="",
        embedding_property="",
    )
    assert index_info is not None
    index_name = index_info.get("name")
    assert index_name == "vector-index-name"
    index_type = index_info.get("type")
    assert index_type == "VECTOR"
    labels_or_types = index_info.get("labelsOrTypes")
    assert labels_or_types == ["Document"]
    properties = index_info.get("properties")
    assert properties == ["vectorProperty"]
    entity_type = index_info.get("entityType")
    assert entity_type == "NODE"
    options = index_info.get("options")
    assert isinstance(options, dict)
    index_config = options.get("indexConfig")
    assert isinstance(index_config, dict)
    embedding_dimension = index_config.get("vector.dimensions")
    assert isinstance(embedding_dimension, int)
    assert embedding_dimension == 1536


@pytest.mark.usefixtures("setup_neo4j_for_retrieval")
def test_retrieve_vector_index_info_wrong_info(driver: neo4j.Driver) -> None:
    index_info = retrieve_vector_index_info(
        driver=driver,
        index_name="err",
        label_or_type="err",
        embedding_property="err",
    )
    assert index_info is None


@pytest.mark.usefixtures("setup_neo4j_for_retrieval")
def test_retrieve_fulltext_index_info_happy_path(driver: neo4j.Driver) -> None:
    index_info = retrieve_fulltext_index_info(
        driver=driver,
        index_name="fulltext-index-name",
        label_or_type="Document",
        text_properties=["vectorProperty"],
    )
    assert index_info is not None
    index_name = index_info.get("name")
    assert index_name == "fulltext-index-name"
    index_type = index_info.get("type")
    assert index_type == "FULLTEXT"
    labels_or_types = index_info.get("labelsOrTypes")
    assert labels_or_types == ["Document"]
    properties = index_info.get("properties")
    assert properties == ["vectorProperty"]
    entity_type = index_info.get("entityType")
    assert entity_type == "NODE"


@pytest.mark.usefixtures("setup_neo4j_for_retrieval")
def test_retrieve_fulltext_index_info_no_index_name(driver: neo4j.Driver) -> None:
    index_info = retrieve_fulltext_index_info(
        driver=driver,
        index_name="",
        label_or_type="Document",
        text_properties=["vectorProperty"],
    )
    assert index_info is not None
    index_name = index_info.get("name")
    assert index_name == "fulltext-index-name"
    index_type = index_info.get("type")
    assert index_type == "FULLTEXT"
    labels_or_types = index_info.get("labelsOrTypes")
    assert labels_or_types == ["Document"]
    properties = index_info.get("properties")
    assert properties == ["vectorProperty"]
    entity_type = index_info.get("entityType")
    assert entity_type == "NODE"


@pytest.mark.usefixtures("setup_neo4j_for_retrieval")
def test_retrieve_fulltext_index_info_no_label_or_properties(
    driver: neo4j.Driver,
) -> None:
    index_info = retrieve_fulltext_index_info(
        driver=driver,
        index_name="fulltext-index-name",
        label_or_type="",
        text_properties=[""],
    )
    assert index_info is not None
    index_name = index_info.get("name")
    assert index_name == "fulltext-index-name"
    index_type = index_info.get("type")
    assert index_type == "FULLTEXT"
    labels_or_types = index_info.get("labelsOrTypes")
    assert labels_or_types == ["Document"]
    properties = index_info.get("properties")
    assert properties == ["vectorProperty"]
    entity_type = index_info.get("entityType")
    assert entity_type == "NODE"


@pytest.mark.usefixtures("setup_neo4j_for_retrieval")
def test_retrieve_fulltext_index_info_wrong_info(driver: neo4j.Driver) -> None:
    index_info = retrieve_fulltext_index_info(
        driver=driver,
        index_name="err",
        label_or_type="err",
        text_properties=[""],
    )
    assert index_info is None


def test_upsert_vectors_on_nodes(driver: neo4j.Driver) -> None:
    driver.execute_query("MATCH (n) DETACH DELETE n;")
    result = driver.execute_query(
        "CREATE (p:Character {name: 'Paul Atreides'}), "
        "(v:Character {name: 'Vladimir Harkonnen'}) "
        "RETURN [elementId(p), elementId(v)] AS ids"
    )
    ids = result.records[0]["ids"]
    assert len(ids) == 2
    upsert_vectors(
        driver=driver,
        ids=ids,
        embedding_property="embedding",
        embeddings=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        entity_type=EntityType.NODE,
    )
    result = driver.execute_query(
        "MATCH (c:Character) RETURN "
        "elementId(c) as id, c.name as name, c.embedding as embedding"
    )
    records = [r.data() for r in result.records]
    records_sorted = sorted(records, key=lambda x: x["id"])
    expected_records = [
        {"id": ids[0], "name": "Paul Atreides", "embedding": [1.0, 2.0, 3.0]},
        {"id": ids[1], "name": "Vladimir Harkonnen", "embedding": [4.0, 5.0, 6.0]},
    ]
    expected_records_sorted = sorted(expected_records, key=lambda x: x["id"])
    assert records_sorted == expected_records_sorted


def test_upsert_vectors_on_relationships(driver: neo4j.Driver) -> None:
    driver.execute_query("MATCH (n) DETACH DELETE n;")
    result = driver.execute_query(
        "CREATE (:Character {name: 'Paul Atreides'})-[a:IS_MEMBER_OF]->(:House {name: 'Atreides'}), "
        "(:Character {name: 'Vladimir Harkonnen'})-[h:IS_MEMBER_OF]->(:House {name: 'Harkonnen'}) "
        "RETURN [elementId(a), elementId(h)] AS ids"
    )
    ids = result.records[0]["ids"]
    assert len(ids) == 2
    upsert_vectors(
        driver=driver,
        ids=ids,
        embedding_property="embedding",
        embeddings=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        entity_type=EntityType.RELATIONSHIP,
    )
    result = driver.execute_query(
        "MATCH (c:Character)-[r:IS_MEMBER_OF]->(h:House) RETURN "
        "type(r) AS type, elementId(r) AS id, r.embedding AS embedding, "
        "c.name as character, h.name AS house"
    )
    records = [r.data() for r in result.records]
    records_sorted = sorted(records, key=lambda x: x["id"])
    expected_records = [
        {
            "type": "IS_MEMBER_OF",
            "id": ids[0],
            "embedding": [1.0, 2.0, 3.0],
            "character": "Paul Atreides",
            "house": "Atreides",
        },
        {
            "type": "IS_MEMBER_OF",
            "id": ids[1],
            "embedding": [4.0, 5.0, 6.0],
            "character": "Vladimir Harkonnen",
            "house": "Harkonnen",
        },
    ]
    expected_records_sorted = sorted(expected_records, key=lambda x: x["id"])
    assert records_sorted == expected_records_sorted
