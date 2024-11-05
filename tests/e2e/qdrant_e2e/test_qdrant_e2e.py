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
from typing import Any, Generator

import pytest
from neo4j import Driver
from neo4j_graphrag.embeddings.base import Embedder
from neo4j_graphrag.embeddings.sentence_transformers import (
    SentenceTransformerEmbeddings,
)
from neo4j_graphrag.retrievers import QdrantNeo4jRetriever
from neo4j_graphrag.types import RetrieverResult, RetrieverResultItem
from qdrant_client import QdrantClient

from ..utils import EMBEDDING_BIOLOGY
from .populate_dbs import populate_dbs


@pytest.fixture(scope="module")
def sentence_transformer_embedder() -> (
    Generator[SentenceTransformerEmbeddings, Any, Any]
):
    embedder = SentenceTransformerEmbeddings(model="all-MiniLM-L6-v2")
    yield embedder


@pytest.fixture(scope="module")
def qdrant_client() -> Generator[Any, Any, Any]:
    client = QdrantClient(url="http://localhost:6333")
    yield client
    client.close()


@pytest.fixture(scope="module")
def populate_qdrant_neo4j(driver: Driver, qdrant_client: QdrantClient) -> None:
    driver.execute_query("MATCH (n) DETACH DELETE n")
    populate_dbs(driver, qdrant_client, "Jeopardy")


@pytest.mark.usefixtures("populate_qdrant_neo4j")
def test_qdrant_neo4j_vector_input(driver: Driver, qdrant_client: QdrantClient) -> None:
    retriever = QdrantNeo4jRetriever(
        driver=driver,
        client=qdrant_client,
        collection_name="Jeopardy",
        id_property_external="neo4j_id",
        id_property_neo4j="id",
    )

    top_k = 1
    results = retriever.search(query_vector=EMBEDDING_BIOLOGY, top_k=top_k)

    assert isinstance(results, RetrieverResult)
    assert len(results.items) == top_k
    assert isinstance(results.items[0], RetrieverResultItem)
    pattern = (
        r"<Record node=<Node element_id='.+' "
        r"labels=frozenset\({'Question'}\) properties={'question': 'In 1953 Watson \& "
        "Crick built a model of the molecular structure of this, the gene-carrying "
        "substance', 'id': 'question_c458c6f64d8d47429636bc5a94c97f51'}> "
        r"score=0.2[0-9]+>"
    )
    assert re.match(pattern, results.items[0].content)


@pytest.mark.usefixtures("populate_qdrant_neo4j")
def test_qdrant_neo4j_text_input_local_embedder(
    driver: Driver,
    qdrant_client: QdrantClient,
    sentence_transformer_embedder: Embedder,
) -> None:
    retriever = QdrantNeo4jRetriever(
        driver=driver,
        client=qdrant_client,
        collection_name="Jeopardy",
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
        r"labels=frozenset\({'Question'}\) properties={'question': 'In 1953 Watson \& "
        "Crick built a model of the molecular structure of this, the gene-carrying "
        "substance', 'id': 'question_c458c6f64d8d47429636bc5a94c97f51'}> "
        r"score=0.2[0-9]+>"
    )
    assert re.match(pattern, results.items[0].content)
