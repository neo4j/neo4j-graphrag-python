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
import os.path
from typing import Any

import neo4j
from neo4j import GraphDatabase
from pinecone import Pinecone, ServerlessSpec

from ..utils import build_data_objects, populate_neo4j

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def populate_dbs(
    neo4j_driver: neo4j.Driver, pc_client: Pinecone, index_name: str = "jeopardy"
) -> None:
    neo4j_objects, pc_question_objs = build_data_objects("pinecone")

    pc_client.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

    # Populate Pinecone
    populate_pinecone(pc_client, pc_question_objs, index_name)

    # Populate Neo4j
    populate_neo4j(neo4j_driver, neo4j_objects)


def populate_pinecone(
    pc_client: Pinecone, pc_question_objs: list[Any], index_name: str
) -> None:
    index = pc_client.Index(index_name)
    index.upsert(vectors=pc_question_objs)


if __name__ == "__main__":
    NEO4J_AUTH = ("neo4j", "password")
    NEO4J_URL = "neo4j://localhost:7687"
    PC_API_KEY = "API_KEY"
    with GraphDatabase.driver(NEO4J_URL, auth=NEO4J_AUTH) as neo4j_driver:
        pc_client = Pinecone(PC_API_KEY)
        populate_dbs(neo4j_driver, pc_client)
