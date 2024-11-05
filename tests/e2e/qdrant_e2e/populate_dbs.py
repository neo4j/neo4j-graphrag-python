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
from __future__ import annotations

from typing import Any

import neo4j
from neo4j import GraphDatabase
from qdrant_client import QdrantClient, models

from ..utils import build_data_objects, populate_neo4j


def populate_dbs(
    neo4j_driver: neo4j.Driver, client: QdrantClient, collection_name: str = "Jeopardy"
) -> None:
    neo4j_objects, question_objs = build_data_objects("qdrant")

    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
    )

    populate_qdrant(client, question_objs, collection_name)

    populate_neo4j(neo4j_driver, neo4j_objects)


def populate_qdrant(
    client: QdrantClient, question_objs: list[Any], collection_name: str
) -> None:
    client.upsert(collection_name=collection_name, points=question_objs)


if __name__ == "__main__":
    NEO4J_AUTH = ("neo4j", "password")
    NEO4J_URL = "neo4j://localhost:7687"
    with GraphDatabase.driver(NEO4J_URL, auth=NEO4J_AUTH) as neo4j_driver:
        populate_dbs(neo4j_driver, QdrantClient(url="http://localhost:6333"))
