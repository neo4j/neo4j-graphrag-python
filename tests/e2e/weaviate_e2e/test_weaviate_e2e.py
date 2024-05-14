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

import pytest
from neo4j import Record
import weaviate
from neo4j_genai.retrievers.external.weaviate import WeaviateNeo4jRetriever
from langchain_openai import OpenAIEmbeddings
from .utils import get_query_vector
from .populate_dbs import populate_dbs
import os


@pytest.fixture(scope="module")
def weaviate_client():
    w_client = weaviate.connect_to_local()
    yield w_client
    w_client.close()


@pytest.fixture(scope="module")
def weaviate_client_openai():
    w_client = weaviate.connect_to_local(
        headers={
            "X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY_E2E_TESTS"),
        }
    )
    yield w_client
    w_client.close()


@pytest.fixture(scope="module")
def populate_weaviate_neo4j(driver, weaviate_client):
    driver.execute_query("MATCH (n) DETACH DELETE n")
    weaviate_client.collections.delete_all()
    populate_dbs(driver, weaviate_client, "Jeopardy")


@pytest.mark.usefixtures("populate_weaviate_neo4j")
def test_weaviate_neo4j_vector_input(driver, weaviate_client):
    retriever = WeaviateNeo4jRetriever(
        driver=driver,
        client=weaviate_client,
        collection="Jeopardy",
        id_property_external="neo4j_id",
        id_property_neo4j="id",
    )

    top_k = 2
    results = retriever.search(query_vector=get_query_vector(), top_k=top_k)

    assert isinstance(results, list)
    assert len(results) == top_k
    assert isinstance(results[0], Record)
    assert results[0].get("node").get("id").startswith("question_")
    assert results[0].get("score") > 0.55


@pytest.mark.usefixtures("populate_weaviate_neo4j")
def test_weaviate_neo4j_text_input_local_embedder(driver, weaviate_client):
    embedder = OpenAIEmbeddings(
        openai_api_key=os.getenv("OPENAI_API_KEY_E2E_TESTS"),
        model="text-embedding-ada-002",
    )
    retriever = WeaviateNeo4jRetriever(
        driver=driver,
        client=weaviate_client,
        collection="Jeopardy",
        id_property_external="neo4j_id",
        id_property_neo4j="id",
        embedder=embedder,
    )

    top_k = 2
    results = retriever.search(query_text="biology", top_k=top_k)

    assert isinstance(results, list)
    assert len(results) == top_k
    assert isinstance(results[0], Record)
    assert results[0].get("node").get("id").startswith("question_")
    assert results[0].get("score") > 0.55


@pytest.mark.usefixtures("populate_weaviate_neo4j")
def test_weaviate_neo4j_text_input_remote_embedder(driver, weaviate_client_openai):
    retriever = WeaviateNeo4jRetriever(
        driver=driver,
        client=weaviate_client_openai,
        collection="Jeopardy",
        id_property_external="neo4j_id",
        id_property_neo4j="id",
    )

    top_k = 2
    results = retriever.search(query_text="biology", top_k=top_k)

    assert isinstance(results, list)
    assert len(results) == top_k
    assert isinstance(results[0], Record)
    assert results[0].get("node").get("id").startswith("question_")
    assert results[0].get("score") > 0.55
