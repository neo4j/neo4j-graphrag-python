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

from neo4j_graphrag.generation.explain import (
    ExplainConfig,
    build_explain_result,
    format_retrieval_context,
    graph_from_retriever,
    sources_from_retriever,
    trace_from_retriever,
)
from neo4j_graphrag.generation.graphrag import GraphRAG
from neo4j_graphrag.llm import LLMResponse
from neo4j_graphrag.types import RetrieverResult, RetrieverResultItem


def test_sources_from_retriever_maps_items() -> None:
    result = RetrieverResult(
        items=[
            RetrieverResultItem(
                content="Movie: Avatar",
                metadata={
                    "score": 0.89,
                    "id": "4:abc",
                    "nodeLabels": ["Movie"],
                    "custom": "keep",
                },
            ),
            RetrieverResultItem(content="plain chunk"),
        ]
    )

    sources = sources_from_retriever(result)

    assert len(sources) == 2
    assert sources[0].index == 1
    assert sources[0].content == "Movie: Avatar"
    assert sources[0].score == 0.89
    assert sources[0].node_id == "4:abc"
    assert sources[0].labels == ["Movie"]
    assert sources[0].metadata == {"custom": "keep"}
    assert sources[1].index == 2
    assert sources[1].score is None


def test_trace_from_retriever_uses_retriever_metadata() -> None:
    result = RetrieverResult(
        items=[],
        metadata={"__retriever": "VectorCypherRetriever"},
    )

    trace = trace_from_retriever(result)

    assert trace.retriever == "VectorCypherRetriever"


def test_trace_from_retriever_defaults_to_unknown() -> None:
    trace = trace_from_retriever(RetrieverResult(items=[]))

    assert trace.retriever == "unknown"


def test_graph_from_retriever_parses_graph_and_paths() -> None:
    result = RetrieverResult(
        items=[
            RetrieverResultItem(
                content="Movie: Avatar",
                metadata={
                    "graph": {
                        "seed_node": {
                            "id": "4:movie",
                            "labels": ["Movie"],
                            "properties": {"title": "Avatar"},
                        },
                        "related_nodes": [
                            {
                                "labels": ["Actor"],
                                "properties": {"name": "Zoe Saldana"},
                            }
                        ],
                        "relationships": [
                            {
                                "type": "ACTED_IN",
                                "start_id": "4:actor",
                                "end_id": "4:movie",
                            }
                        ],
                        "paths": [
                            [
                                {
                                    "labels": ["Actor"],
                                    "properties": {"name": "Zoe Saldana"},
                                },
                                {"type": "ACTED_IN"},
                                {
                                    "labels": ["Movie"],
                                    "properties": {"title": "Avatar"},
                                },
                            ]
                        ],
                    }
                },
            )
        ]
    )

    graph = graph_from_retriever(result)

    assert graph is not None
    assert len(graph) == 1
    assert graph[0].seed_node is not None
    assert graph[0].seed_node.properties["title"] == "Avatar"
    assert graph[0].related_nodes[0].properties["name"] == "Zoe Saldana"
    assert graph[0].relationships[0].type == "ACTED_IN"
    assert len(graph[0].paths) == 1
    assert graph[0].paths[0][1].type == "ACTED_IN"


def test_graph_from_retriever_returns_none_without_graph_metadata() -> None:
    result = RetrieverResult(
        items=[RetrieverResultItem(content="chunk", metadata={"score": 0.5})]
    )

    assert graph_from_retriever(result) is None


def test_build_explain_result_combines_sources_trace_and_graph() -> None:
    result = RetrieverResult(
        items=[
            RetrieverResultItem(
                content="Movie: Avatar",
                metadata={
                    "score": 0.89,
                    "graph": {
                        "seed_node": {
                            "labels": ["Movie"],
                            "properties": {"title": "Avatar"},
                        }
                    },
                },
            )
        ],
        metadata={"__retriever": "VectorCypherRetriever"},
    )

    explain = build_explain_result(result)

    assert explain.trace.retriever == "VectorCypherRetriever"
    assert len(explain.sources) == 1
    assert explain.graph is not None
    assert explain.graph[0].seed_node is not None


def test_format_retrieval_context_adds_source_indexes() -> None:
    result = RetrieverResult(
        items=[
            RetrieverResultItem(content="first"),
            RetrieverResultItem(content="second"),
        ]
    )

    context = format_retrieval_context(result, cite_sources=True)

    assert context == "[1] first\n[2] second"


def test_graphrag_search_attaches_explain(
    retriever_mock: MagicMock, llm: MagicMock
) -> None:
    rag = GraphRAG(retriever=retriever_mock, llm=llm)
    retriever_mock.search.return_value = RetrieverResult(
        items=[RetrieverResultItem(content="Movie: Avatar", metadata={"score": 0.9})],
        metadata={"__retriever": "VectorCypherRetriever"},
    )
    llm.invoke.return_value = LLMResponse(content="answer [1]")

    result = rag.search("Who acted in Avatar?", explain=ExplainConfig())

    assert result.explain is not None
    assert result.explain.trace.retriever == "VectorCypherRetriever"
    assert result.retriever_result is not None
    llm.invoke.assert_called_once()
    assert "[1] Movie: Avatar" in llm.invoke.call_args.kwargs["input"]
    assert "Cite sources inline" in llm.invoke.call_args.kwargs["system_instruction"]


def test_graphrag_search_without_explain_unchanged(
    retriever_mock: MagicMock, llm: MagicMock
) -> None:
    rag = GraphRAG(retriever=retriever_mock, llm=llm)
    retriever_mock.search.return_value = RetrieverResult(
        items=[RetrieverResultItem(content="chunk")]
    )
    llm.invoke.return_value = LLMResponse(content="answer")

    result = rag.search("question")

    assert result.explain is None
    assert result.retriever_result is None
    assert "chunk" in llm.invoke.call_args.kwargs["input"]
    assert "[1]" not in llm.invoke.call_args.kwargs["input"]
