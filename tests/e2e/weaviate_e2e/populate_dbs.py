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

import weaviate.classes as wvc
from weaviate.client import Client
from weaviate.connect.helpers import connect_to_local
from typing import Any
from neo4j import GraphDatabase, Driver, EagerResult
import os.path
import json
import hashlib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def populate_dbs(
    neo4j_driver: Driver, w_client: Client, collection_name: str = "Jeopardy"
) -> None:
    neo4j_objects, w_question_objs = build_data_objects()
    w_client.collections.create(
        collection_name,
        vectorizer_config=wvc.config.Configure.Vectorizer.text2vec_transformers(),
        vector_index_config=wvc.config.Configure.VectorIndex.hnsw(
            distance_metric=wvc.config.VectorDistances.COSINE  # select prefered distance metric
        ),
        properties=[
            wvc.config.Property(name="neo4j_id", data_type=wvc.config.DataType.TEXT),
        ],
    )

    # Populate Weaviate
    populate_weaviate(w_client, w_question_objs, collection_name)

    # Populate Neo4j
    populate_neo4j(neo4j_driver, neo4j_objects)


def populate_neo4j(
    neo4j_driver: Driver, neo4j_objs: dict[str, list[wvc.data.DataObject]]
) -> EagerResult:
    question_nodes = list(
        filter(lambda x: x["label"] == "Question", neo4j_objs["nodes"])
    )
    answer_nodes = list(filter(lambda x: x["label"] == "Answer", neo4j_objs["nodes"]))
    category_nodes = list(
        filter(lambda x: x["label"] == "Category", neo4j_objs["nodes"])
    )
    belongs_to_relationships = list(
        filter(lambda x: x["type"] == "BELONGS_TO", neo4j_objs["relationships"])
    )
    has_answer_relationships = list(
        filter(lambda x: x["type"] == "HAS_ANSWER", neo4j_objs["relationships"])
    )
    question_nodes_cypher = "UNWIND $nodes as node MERGE (n:Question {id: node.properties.id}) ON CREATE SET n = node.properties"
    answer_nodes_cypher = "UNWIND $nodes as node MERGE (n:Answer {id: node.properties.id}) ON CREATE SET n = node.properties"
    category_nodes_cypher = (
        "UNWIND $nodes as node MERGE (n:Category {id: node.id}) ON CREATE SET n = node"
    )
    belongs_to_relationships_cypher = "UNWIND $relationships as rel MATCH (q:Question {id: rel.start_node_id}), (c:Category {id: rel.end_node_id}) MERGE (q)-[r:BELONGS_TO]->(c)"
    has_answer_relationships_cypher = "UNWIND $relationships as rel MATCH (q:Question {id: rel.start_node_id}), (a:Answer {id: rel.end_node_id}) MERGE (q)-[r:HAS_ANSWER]->(a)"
    neo4j_driver.execute_query(question_nodes_cypher, {"nodes": question_nodes})
    neo4j_driver.execute_query(answer_nodes_cypher, {"nodes": answer_nodes})
    neo4j_driver.execute_query(category_nodes_cypher, {"nodes": category_nodes})
    neo4j_driver.execute_query(
        belongs_to_relationships_cypher, {"relationships": belongs_to_relationships}
    )
    res = neo4j_driver.execute_query(
        has_answer_relationships_cypher, {"relationships": has_answer_relationships}
    )
    return res


def populate_weaviate(
    w_client: Client,
    w_question_objs: list[wvc.data.DataObject],
    collection_name: str,
) -> None:
    questions = w_client.collections.get(collection_name)
    questions.data.insert_many(w_question_objs)


def build_data_objects() -> tuple[dict[str, Any], list[wvc.data.DataObject]]:
    # read file from disk
    # this file is from https://github.com/weaviate-tutorials/quickstart/tree/main/data
    # MIT License
    file_name = os.path.join(
        BASE_DIR,
        "../data/jeopardy_tiny_with_vectors_all-MiniLM-L6-v2.json",
    )
    with open(file_name, "r") as f:
        data = json.load(f)

    w_question_objs = list()
    neo4j_objs: dict[str, Any] = {"nodes": [], "relationships": []}

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
        w_question_objs.append(
            wvc.data.DataObject(
                properties={
                    "neo4j_id": f"question_{id}",
                },
                vector=d["vector"],
            )
        )

    return neo4j_objs, w_question_objs


if __name__ == "__main__":
    neo4j_auth = ("neo4j", "password")
    neo4j_url = "neo4j://localhost:7687"
    with GraphDatabase.driver(neo4j_url, auth=neo4j_auth) as neo4j_driver:
        with connect_to_local() as w_client:
            populate_dbs(neo4j_driver, w_client)
