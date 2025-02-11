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
    upsert_texts_and_embeddings,
)


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


def test_upsert_texts_and_embeddings(driver: neo4j.Driver) -> None:
    driver.execute_query("MATCH (n) DETACH DELETE n")
    upsert_texts_and_embeddings(
        driver=driver,
        node_label="Character",
        text_property="name",
        embedding_property="vectorProperty",
        texts=["Paul Atreides", "Chani", "Vladimir Harkonnen"],
        embeddings=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        neo4j_version_is_5_23_or_above=True,
        metadatas=[{"house": "Atreides"}, {"house": "Fremen"}, {"house": "Harkonnen"}],
        ids=["1", "2", "3"],
    )
    result = driver.execute_query(
        (
            "MATCH (n:Character) "
            "RETURN n.id as id, n.name as name, n.house as house, "
            "n.vectorProperty as vectorProperty;"
        )
    )
    records = [r.data() for r in result.records]
    expected_records = [
        {
            "house": "Atreides",
            "id": "1",
            "name": "Paul Atreides",
            "vectorProperty": [1.0, 2.0, 3.0],
        },
        {
            "house": "Fremen",
            "id": "2",
            "name": "Chani",
            "vectorProperty": [4.0, 5.0, 6.0],
        },
        {
            "house": "Harkonnen",
            "id": "3",
            "name": "Vladimir Harkonnen",
            "vectorProperty": [7.0, 8.0, 9.0],
        },
    ]
    assert sorted(records, key=lambda x: x["id"]) == sorted(
        expected_records,
        key=lambda x: x["id"],  # type: ignore
    )
