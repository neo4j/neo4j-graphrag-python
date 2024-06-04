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

import re
import pytest
from typing import Generator, Any
from neo4j import Driver

from weaviate.client import Client
from weaviate.connect.helpers import connect_to_local
from neo4j_genai.embedder import Embedder
from neo4j_genai.retrievers.external.weaviate import WeaviateNeo4jRetriever
from neo4j_genai import WeaviateNeo4jRetriever
from langchain_community.embeddings import HuggingFaceEmbeddings

from neo4j_genai.types import RetrieverResult, RetrieverResultItem
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

    assert isinstance(results, RetrieverResult)
    assert len(results.items) == top_k
    assert isinstance(results.items[0], RetrieverResultItem)
    pattern = (
        r"<Record node=<Node element_id='.+' "
        "labels=frozenset\({'Question'}\) properties={'question': 'In 1953 Watson \& "
        "Crick built a model of the molecular structure of this, the gene-carrying "
        "substance', 'id': 'question_c458c6f64d8d47429636bc5a94c97f51'}> "
        r"score=0.6[0-9]+>"
    )
    assert re.match(pattern, results.items[0].content)


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

    assert isinstance(results, RetrieverResult)
    assert len(results.items) == top_k
    assert isinstance(results.items[0], RetrieverResultItem)
    pattern = (
        r"<Record node=<Node element_id='.+' "
        "labels=frozenset\({'Question'}\) properties={'question': 'In 1953 Watson \& "
        "Crick built a model of the molecular structure of this, the gene-carrying "
        "substance', 'id': 'question_c458c6f64d8d47429636bc5a94c97f51'}> "
        r"score=0.6[0-9]+>"
    )
    assert re.match(pattern, results.items[0].content)


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

    assert isinstance(results, RetrieverResult)
    assert len(results.items) == top_k
    assert isinstance(results.items[0], RetrieverResultItem)
    pattern = (
        r"<Record node=<Node element_id='.+' "
        "labels=frozenset\({'Question'}\) properties={'question': 'In 1953 Watson \& "
        "Crick built a model of the molecular structure of this, the gene-carrying "
        "substance', 'id': 'question_c458c6f64d8d47429636bc5a94c97f51'}> "
        r"score=0.5[0-9]+>"
    )
    assert re.match(pattern, results.items[0].content)
