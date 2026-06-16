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
from neo4j_graphrag.generation.explain import (
    ExplainConfig,
    ExplainResult,
    GraphContext,
    GraphRelationshipRef,
    TraceStep,
    build_explain_result,
    format_retrieval_context,
    graph_from_retriever,
    serialize_neo4j_path,
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
    assert trace.cypher is None


def test_trace_from_retriever_includes_generated_cypher() -> None:
    result = RetrieverResult(
        items=[],
        metadata={
            "__retriever": "Text2CypherRetriever",
            "cypher": "MATCH (m:Movie) RETURN m.title AS title",
        },
    )

    trace = trace_from_retriever(result)

    assert trace.retriever == "Text2CypherRetriever"
    assert trace.cypher == "MATCH (m:Movie) RETURN m.title AS title"


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


def test_graphrag_search_with_explain_skips_llm_when_no_rows(
    retriever_mock: MagicMock, llm: MagicMock
) -> None:
    rag = GraphRAG(retriever=retriever_mock, llm=llm)
    retriever_mock.search.return_value = RetrieverResult(
        items=[],
        metadata={
            "__retriever": "Text2CypherRetriever",
            "cypher": "MATCH (m:Movie) RETURN m.title AS title",
        },
    )

    result = rag.search("question", explain=ExplainConfig())

    assert "could not find any matching results" in result.answer.lower()
    assert result.explain is not None
    assert result.explain.trace.cypher == "MATCH (m:Movie) RETURN m.title AS title"
    assert result.explain.sources == []
    llm.invoke.assert_not_called()


def test_serialize_neo4j_path_from_graph_objects() -> None:
    actor = MagicMock(spec=neo4j.graph.Node)
    actor.element_id = "4:actor"
    actor.labels = ["Actor"]
    actor.items.return_value = {"name": "Zoe Saldana"}.items()
    movie = MagicMock(spec=neo4j.graph.Node)
    movie.element_id = "4:movie"
    movie.labels = ["Movie"]
    movie.items.return_value = {"title": "Avatar"}.items()
    relationship = MagicMock(spec=neo4j.graph.Relationship)
    relationship.type = "ACTED_IN"
    relationship.start_node = actor
    relationship.end_node = movie
    path = MagicMock(spec=neo4j.graph.Path)
    path.nodes = [actor, movie]
    path.relationships = [relationship]

    serialized = serialize_neo4j_path(path)

    assert serialized[0].id == "4:actor"
    assert serialized[0].properties["name"] == "Zoe Saldana"
    assert serialized[1].type == "ACTED_IN"
    assert serialized[1].start_id == "4:actor"
    assert serialized[1].end_id == "4:movie"
    assert serialized[2].properties["title"] == "Avatar"


def test_node_from_neo4j_graph_node_serializes_temporal_properties() -> None:
    import neo4j.time

    movie = MagicMock(spec=neo4j.graph.Node)
    movie.element_id = "4:movie"
    movie.labels = ["Movie"]
    movie.items.return_value = {
        "title": "One Flew Over the Cuckoo's Nest",
        "released": neo4j.time.Date(1975, 11, 19),
    }.items()

    from neo4j_graphrag.generation.explain import _node_from_neo4j_graph_node

    node_ref = _node_from_neo4j_graph_node(movie)

    assert node_ref.properties["released"] == "1975-11-19"
    assert ExplainResult(
        sources=[],
        trace=TraceStep(retriever="VectorCypherRetriever"),
        graph=[
            GraphContext(
                seed_node=node_ref,
                paths=[[node_ref, GraphRelationshipRef(type="ACTED_IN"), node_ref]],
            )
        ],
    ).model_dump(mode="json")


def test_text2cypher_explain_result_formatter_formats_record() -> None:
    from neo4j_graphrag.generation.explain import text2cypher_explain_result_formatter

    record = neo4j.Record({"title": "One Flew Over the Cuckoo's Nest"})

    item = text2cypher_explain_result_formatter(record)

    assert item.content == "title: One Flew Over the Cuckoo's Nest"
    assert item.metadata is not None
    assert item.metadata["graph"]["seed_node"]["properties"]["title"] == (
        "One Flew Over the Cuckoo's Nest"
    )


def test_text2cypher_explain_result_formatter_includes_graph_paths() -> None:
    from neo4j_graphrag.generation.explain import text2cypher_explain_result_formatter

    director = MagicMock(spec=neo4j.graph.Node)
    director.element_id = "4:director"
    director.labels = ["Person"]
    director.items.return_value = {"name": "Joel Coen"}.items()
    actor = MagicMock(spec=neo4j.graph.Node)
    actor.element_id = "4:actor"
    actor.labels = ["Actor"]
    actor.items.return_value = {"name": "Steve Buscemi"}.items()
    movie = MagicMock(spec=neo4j.graph.Node)
    movie.element_id = "4:movie"
    movie.labels = ["Movie"]
    movie.items.return_value = {"title": "Fargo"}.items()
    directed = MagicMock(spec=neo4j.graph.Relationship)
    directed.type = "DIRECTED"
    directed.start_node = director
    directed.end_node = movie
    acted = MagicMock(spec=neo4j.graph.Relationship)
    acted.type = "ACTED_IN"
    acted.start_node = actor
    acted.end_node = movie
    path = MagicMock(spec=neo4j.graph.Path)
    path.nodes = [director, movie, actor]
    path.relationships = [directed, acted]
    record = neo4j.Record({"title": "Fargo", "path": path})

    item = text2cypher_explain_result_formatter(record)

    assert item.content == "title: Fargo"
    assert item.metadata is not None
    assert len(item.metadata["graph"]["paths"]) == 1
    assert item.metadata["graph"]["paths"][0][1]["type"] == "DIRECTED"
