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
from neo4j_graphrag.generation.explain import GraphRelationshipRef, build_explain_result
from neo4j_graphrag.types import RetrieverResult
from graphrag_with_explain import (
    graph_and_paths_from_record,
    movies_vector_cypher_explain_formatter,
)


def test_graph_and_paths_from_record_builds_seed_actors_and_directors() -> None:
    record = neo4j.Record(
        {
            "movieTitle": "One Flew Over the Cuckoo's Nest",
            "moviePlot": "A criminal pleads insanity.",
            "actors": ["Michael Berryman"],
            "directors": ["Milos Forman"],
            "paths": None,
        }
    )

    graph = graph_and_paths_from_record(record)

    assert (
        graph["seed_node"]["properties"]["title"] == "One Flew Over the Cuckoo's Nest"
    )
    assert len(graph["related_nodes"]) == 2
    assert len(graph["paths"]) == 2
    assert graph["paths"][0][1]["type"] == "ACTED_IN"
    assert graph["paths"][1][1]["type"] == "DIRECTED"


def test_graph_and_paths_from_record_builds_seed_and_actors() -> None:
    record = neo4j.Record(
        {
            "movieTitle": "Avatar",
            "moviePlot": "A marine on an alien planet.",
            "actors": ["Zoe Saldana", "Sam Worthington"],
            "paths": None,
        }
    )

    graph = graph_and_paths_from_record(record)

    assert graph["seed_node"]["properties"]["title"] == "Avatar"
    assert len(graph["related_nodes"]) == 2
    assert graph["related_nodes"][0]["properties"]["name"] == "Zoe Saldana"
    assert len(graph["paths"]) == 2
    assert graph["paths"][0][1]["type"] == "ACTED_IN"


def test_graph_and_paths_from_record_uses_neo4j_paths() -> None:
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
    record = neo4j.Record(
        {
            "node": movie,
            "movieTitle": "Avatar",
            "moviePlot": "A marine on an alien planet.",
            "actors": ["Zoe Saldana"],
            "paths": [path],
        }
    )

    graph = graph_and_paths_from_record(record)

    assert graph["seed_node"]["id"] == "4:movie"
    assert len(graph["paths"]) == 1
    assert graph["paths"][0][0]["id"] == "4:actor"
    assert graph["paths"][0][1]["start_id"] == "4:actor"


def test_movies_vector_cypher_explain_formatter_attaches_graph() -> None:
    record = neo4j.Record(
        {
            "movieTitle": "Avatar",
            "moviePlot": "A marine on an alien planet.",
            "actors": ["Zoe Saldana"],
            "directors": ["James Cameron"],
            "similarityScore": 0.91,
            "paths": None,
        }
    )

    item = movies_vector_cypher_explain_formatter(record)

    assert "Avatar" in item.content
    assert "James Cameron" in item.content
    assert item.metadata is not None
    assert item.metadata["score"] == 0.91
    assert item.metadata["graph"]["seed_node"]["properties"]["title"] == "Avatar"


def test_build_explain_result_from_movies_formatter_output() -> None:
    record = neo4j.Record(
        {
            "movieTitle": "Avatar",
            "moviePlot": "A marine on an alien planet.",
            "actors": ["Zoe Saldana"],
            "similarityScore": 0.91,
            "paths": None,
        }
    )
    result = RetrieverResult(
        items=[movies_vector_cypher_explain_formatter(record)],
        metadata={"__retriever": "VectorCypherRetriever"},
    )

    explain = build_explain_result(result)

    assert explain.graph is not None
    path_rel = explain.graph[0].paths[0][1]
    assert isinstance(path_rel, GraphRelationshipRef)
    assert path_rel.type == "ACTED_IN"
    assert explain.sources[0].score == 0.91
