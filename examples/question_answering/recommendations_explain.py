"""VectorCypher explain helpers for the public recommendations demo database.

Used by ``graphrag_with_explain.py``; copy or adapt for your own graph schema.
"""

from __future__ import annotations

from typing import Any

import neo4j

from neo4j_graphrag.generation.explain import (
    GraphContext,
    GraphNodeRef,
    GraphRelationshipRef,
    node_from_neo4j_graph_node,
    serialize_paths,
    vector_cypher_explain_result_formatter,
)
from neo4j_graphrag.types import RetrieverResultItem

MOVIES_ACTORS_PATH_RETRIEVAL_QUERY = """
MATCH path = (actor:Actor)-[:ACTED_IN]->(node)
WITH node, score, collect(DISTINCT actor.name) AS actors, collect(path) AS actorPaths
OPTIONAL MATCH directorPath = (director:Person)-[:DIRECTED]->(node)
WITH node, score, actors, actorPaths,
     [name IN collect(DISTINCT director.name) WHERE name IS NOT NULL] AS directors,
     [path IN collect(directorPath) WHERE path IS NOT NULL] AS directorPaths
RETURN node.title AS movieTitle,
       node.plot AS moviePlot,
       actors,
       directors,
       actorPaths + directorPaths AS paths,
       score AS similarityScore
""".strip()


def graph_and_paths_from_record(
    record: neo4j.Record,
    *,
    node_key: str = "node",
    title_key: str = "movieTitle",
    plot_key: str = "moviePlot",
    actors_key: str = "actors",
    directors_key: str = "directors",
    paths_key: str = "paths",
    movie_labels: list[str] | None = None,
    actor_labels: list[str] | None = None,
    director_labels: list[str] | None = None,
) -> dict[str, Any]:
    """Build a metadata.graph mapping from a recommendations VectorCypher row."""
    movie_label_list = movie_labels or ["Movie"]
    actor_label_list = actor_labels or ["Actor"]
    director_label_list = director_labels or ["Person"]

    seed_node: GraphNodeRef | None = None
    node = record.get(node_key)
    if isinstance(node, neo4j.graph.Node):
        seed_node = node_from_neo4j_graph_node(node)
    else:
        title = record.get(title_key)
        plot = record.get(plot_key)
        if title is not None or plot is not None:
            properties: dict[str, Any] = {}
            if title is not None:
                properties["title"] = title
            if plot is not None:
                properties["plot"] = plot
            seed_node = GraphNodeRef(labels=movie_label_list, properties=properties)

    related_nodes: list[GraphNodeRef] = []
    relationships: list[GraphRelationshipRef] = []
    actors = record.get(actors_key) or []
    if isinstance(actors, list):
        for actor_name in actors:
            if actor_name is None:
                continue
            actor_node = GraphNodeRef(
                labels=actor_label_list,
                properties={"name": str(actor_name)},
            )
            related_nodes.append(actor_node)
            if seed_node is not None and seed_node.id is not None:
                relationships.append(
                    GraphRelationshipRef(
                        type="ACTED_IN",
                        start_id=None,
                        end_id=seed_node.id,
                    )
                )

    directors = record.get(directors_key) or []
    if isinstance(directors, list):
        for director_name in directors:
            if director_name is None:
                continue
            director_node = GraphNodeRef(
                labels=director_label_list,
                properties={"name": str(director_name)},
            )
            related_nodes.append(director_node)
            if seed_node is not None and seed_node.id is not None:
                relationships.append(
                    GraphRelationshipRef(
                        type="DIRECTED",
                        start_id=None,
                        end_id=seed_node.id,
                    )
                )

    paths = serialize_paths(record.get(paths_key))
    if not paths and seed_node is not None and related_nodes:
        for person_name, rel_type, labels in (
            *(
                (name, "ACTED_IN", actor_label_list)
                for name in actors
                if isinstance(actors, list) and name is not None
            ),
            *(
                (name, "DIRECTED", director_label_list)
                for name in directors
                if isinstance(directors, list) and name is not None
            ),
        ):
            person_node = GraphNodeRef(
                labels=labels,
                properties={"name": str(person_name)},
            )
            paths.append(
                [
                    person_node,
                    GraphRelationshipRef(type=rel_type),
                    seed_node,
                ]
            )

    return GraphContext(
        seed_node=seed_node,
        related_nodes=related_nodes,
        relationships=relationships,
        paths=paths,
    ).model_dump(exclude_none=True)


def movies_vector_cypher_explain_formatter(
    record: neo4j.Record,
) -> RetrieverResultItem:
    actors = record.get("actors") or []
    if isinstance(actors, list):
        actors_text = ", ".join(str(actor) for actor in actors if actor is not None)
    else:
        actors_text = str(actors)
    directors = record.get("directors") or []
    if isinstance(directors, list):
        directors_text = ", ".join(
            str(director) for director in directors if director is not None
        )
    else:
        directors_text = str(directors)
    title = record.get("movieTitle")
    plot = record.get("moviePlot")
    content = (
        f"Movie title: {title}, Plot: {plot}, "
        f"Actors: {actors_text}, Directors: {directors_text}"
    )
    return vector_cypher_explain_result_formatter(
        record,
        content=content,
        graph_builder=graph_and_paths_from_record,
    )
