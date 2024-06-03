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
from typing import Generator, Any
from neo4j import Record, Driver

from weaviate.client import Client
from weaviate.connect.helpers import connect_to_local
from neo4j_genai.embedder import Embedder
from neo4j_genai.retrievers.external.weaviate import WeaviateNeo4jRetriever
from langchain_community.embeddings import HuggingFaceEmbeddings
from .utils import EMBEDDING_BIOLOGY
from .populate_dbs import populate_dbs


@pytest.fixture(scope="module")
def sentence_transformer_embedder() -> Generator[HuggingFaceEmbeddings, Any, Any]:
    embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    yield embedder


@pytest.fixture(scope="module")
def weaviate_client() -> Generator[Any, Any, Any]:
    w_client = connect_to_local()
    yield w_client
    w_client.close()


@pytest.fixture(scope="module")
def populate_weaviate_neo4j(driver: Driver, weaviate_client: Client) -> None:
    driver.execute_query("MATCH (n) DETACH DELETE n")
    weaviate_client.collections.delete_all()
    populate_dbs(driver, weaviate_client, "Jeopardy")


@pytest.mark.usefixtures("populate_weaviate_neo4j")
def test_weaviate_neo4j_vector_input(driver: Driver, weaviate_client: Client) -> None:
    retriever = WeaviateNeo4jRetriever(
        driver=driver,
        client=weaviate_client,
        collection="Jeopardy",
        id_property_external="neo4j_id",
        id_property_neo4j="id",
    )

    top_k = 2
    results = retriever.search(query_vector=EMBEDDING_BIOLOGY, top_k=top_k)

    assert isinstance(results, list)
    assert len(results) == top_k
    assert isinstance(results[0], Record)
    assert results[0].get("node").get("id").startswith("question_")
    assert results[0].get("score") > 0.55


@pytest.mark.usefixtures("populate_weaviate_neo4j")
def test_weaviate_neo4j_text_input_local_embedder(
    driver: Driver,
    weaviate_client: Client,
    sentence_transformer_embedder: Embedder,
) -> None:
    retriever = WeaviateNeo4jRetriever(
        driver=driver,
        client=weaviate_client,
        collection="Jeopardy",
        id_property_external="neo4j_id",
        id_property_neo4j="id",
        embedder=sentence_transformer_embedder,
    )

    top_k = 2
    results = retriever.search(query_text="biology", top_k=top_k)

    assert isinstance(results, list)
    assert len(results) == top_k
    assert isinstance(results[0], Record)
    assert results[0].get("node").get("id").startswith("question_")
    assert results[0].get("score") > 0.55


@pytest.mark.usefixtures("populate_weaviate_neo4j")
def test_weaviate_neo4j_text_input_remote_embedder(
    driver: Driver, weaviate_client: Client
) -> None:
    retriever = WeaviateNeo4jRetriever(
        driver=driver,
        client=weaviate_client,
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
