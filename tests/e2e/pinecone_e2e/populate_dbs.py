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

import hashlib
import json
import os.path

from neo4j import GraphDatabase
from pinecone import Pinecone, ServerlessSpec

from ..utils import populate_neo4j

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def populate_dbs(neo4j_driver, pc_client, index_name="jeopardy"):
    neo4j_objects, pc_question_objs = build_data_objects()

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


def populate_pinecone(pc_client, pc_question_objs, index_name):
    index = pc_client.Index(index_name)
    index.upsert(vectors=pc_question_objs)


def build_data_objects():
    # read file from disk
    # this file is from https://github.com/weaviate-tutorials/quickstart/tree/main/data
    # MIT License
    file_name = os.path.join(
        BASE_DIR,
        "../data/jeopardy_tiny_with_vectors_all-MiniLM-L6-v2.json",
    )
    with open(file_name, "r") as f:
        data = json.load(f)

    pc_question_objs = list()
    neo4j_objs = {"nodes": [], "relationships": []}

    # only unique categories and ids for them
    unique_categories_list = list(set([c["Category"] for c in data]))
    unique_categories = [
        {"label": "Category", "name": c, "id": c} for c in unique_categories_list
    ]
    neo4j_objs["nodes"] += unique_categories

    for i, d in enumerate(data):
        id = hashlib.md5(d["Question"].encode()).hexdigest()
        neo4j_objs["nodes"].append(
            {
                "label": "Question",
                "properties": {
                    "id": f"question_{id}",
                    "question": d["Question"],
                },
            }
        )
        neo4j_objs["nodes"].append(
            {
                "label": "Answer",
                "properties": {
                    "id": f"answer_{id}",
                    "answer": d["Answer"],
                },
            }
        )
        neo4j_objs["relationships"].append(
            {
                "start_node_id": f"question_{id}",
                "end_node_id": f"answer_{id}",
                "type": "HAS_ANSWER",
                "properties": {},
            }
        )
        neo4j_objs["relationships"].append(
            {
                "start_node_id": f"question_{id}",
                "end_node_id": d["Category"],
                "type": "BELONGS_TO",
                "properties": {},
            }
        )
        pc_question_objs.append({"id": f"question_{id}", "values": d["vector"]})

    return neo4j_objs, pc_question_objs


if __name__ == "__main__":
    NEO4J_AUTH = ("neo4j", "password")
    NEO4J_URL = "neo4j://localhost:7687"
    PC_API_KEY = "API_KEY"
    with GraphDatabase.driver(NEO4J_URL, auth=NEO4J_AUTH) as neo4j_driver:
        pc_client = Pinecone(PC_API_KEY)
        populate_dbs(neo4j_driver, pc_client)
