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

from unittest.mock import MagicMock

import neo4j
import pytest
from neo4j_graphrag.exceptions import LLMGenerationError
from neo4j_graphrag.generation.graphrag import GraphRAG
from neo4j_graphrag.generation.types import RagResultModel
from neo4j_graphrag.llm import LLMResponse
from neo4j_graphrag.retrievers import VectorCypherRetriever
from neo4j_graphrag.types import RetrieverResult

from tests.e2e.conftest import BiologyEmbedder
from tests.e2e.utils import build_data_objects, populate_neo4j


@pytest.fixture(scope="module")
def populate_neo4j_db(driver: neo4j.Driver) -> None:
    driver.execute_query("MATCH (n) DETACH DELETE n")
    neo4j_objects, q_objects = build_data_objects(q_vector_fmt="neo4j")
    populate_neo4j(driver, neo4j_objects, should_create_vector_index=True)


@pytest.mark.usefixtures("populate_neo4j_db")
def test_graphrag_happy_path(
    driver: MagicMock, llm: MagicMock, biology_embedder: BiologyEmbedder
) -> None:
    retriever = VectorCypherRetriever(
        driver,
        retrieval_query="WITH node RETURN node {.question}",
        index_name="vector-index-name",
        embedder=biology_embedder,
    )
    rag = GraphRAG(
        retriever=retriever,
        llm=llm,
    )
    llm.invoke.return_value = LLMResponse(content="some text")

    result = rag.search(
        query_text="biology",
        retriever_config={
            "top_k": 2,
        },
    )

    llm.invoke.assert_called_once_with(
        """Context:
<Record node={'question': 'In 1953 Watson & Crick built a model of the molecular structure of this, the gene-carrying substance'}>
<Record node={'question': 'This organ removes excess glucose from the blood & stores it as glycogen'}>

Examples:


Question:
biology

Answer:
""",
        None,
        system_instruction="Answer the user question using the provided context.",
    )
    assert isinstance(result, RagResultModel)
    assert result.answer == "some text"
    assert result.retriever_result is None


@pytest.mark.usefixtures("populate_neo4j_db")
def test_graphrag_happy_path_return_context(
    driver: MagicMock, llm: MagicMock, biology_embedder: BiologyEmbedder
) -> None:
    retriever = VectorCypherRetriever(
        driver,
        retrieval_query="WITH node RETURN node {.question}",
        index_name="vector-index-name",
        embedder=biology_embedder,
    )
    rag = GraphRAG(
        retriever=retriever,
        llm=llm,
    )
    llm.invoke.return_value = LLMResponse(content="some text")

    result = rag.search(
        query_text="biology",
        retriever_config={
            "top_k": 2,
        },
        return_context=True,
    )

    llm.invoke.assert_called_once_with(
        """Context:
<Record node={'question': 'In 1953 Watson & Crick built a model of the molecular structure of this, the gene-carrying substance'}>
<Record node={'question': 'This organ removes excess glucose from the blood & stores it as glycogen'}>

Examples:


Question:
biology

Answer:
""",
        None,
        system_instruction="Answer the user question using the provided context.",
    )
    assert isinstance(result, RagResultModel)
    assert result.answer == "some text"
    assert isinstance(result.retriever_result, RetrieverResult)
    assert len(result.retriever_result.items) == 2


@pytest.mark.usefixtures("populate_neo4j_db")
def test_graphrag_happy_path_examples(
    driver: MagicMock, llm: MagicMock, biology_embedder: BiologyEmbedder
) -> None:
    retriever = VectorCypherRetriever(
        driver,
        retrieval_query="WITH node RETURN node {.question}",
        index_name="vector-index-name",
        embedder=biology_embedder,
    )
    rag = GraphRAG(
        retriever=retriever,
        llm=llm,
    )
    llm.invoke.return_value = LLMResponse(content="some text")

    result = rag.search(
        query_text="biology",
        retriever_config={
            "top_k": 2,
        },
        examples="this is my example",
    )

    llm.invoke.assert_called_once_with(
        """Context:
<Record node={'question': 'In 1953 Watson & Crick built a model of the molecular structure of this, the gene-carrying substance'}>
<Record node={'question': 'This organ removes excess glucose from the blood & stores it as glycogen'}>

Examples:
this is my example

Question:
biology

Answer:
""",
        None,
        system_instruction="Answer the user question using the provided context.",
    )
    assert result.answer == "some text"


@pytest.mark.usefixtures("populate_neo4j_db")
def test_graphrag_llm_error(
    driver: MagicMock, llm: MagicMock, biology_embedder: BiologyEmbedder
) -> None:
    retriever = VectorCypherRetriever(
        driver,
        retrieval_query="WITH node RETURN node {.question}",
        index_name="vector-index-name",
        embedder=biology_embedder,
    )
    rag = GraphRAG(
        retriever=retriever,
        llm=llm,
    )
    llm.invoke.side_effect = LLMGenerationError("error")

    with pytest.raises(LLMGenerationError):
        rag.search(
            query_text="biology",
        )


@pytest.mark.usefixtures("populate_neo4j_db")
def test_graphrag_retrieval_error(
    driver: MagicMock, llm: MagicMock, retriever_mock: MagicMock
) -> None:
    rag = GraphRAG(
        retriever=retriever_mock,
        llm=llm,
    )

    retriever_mock.search.side_effect = TypeError("error")

    with pytest.raises(TypeError):
        rag.search(
            query_text="biology",
        )
