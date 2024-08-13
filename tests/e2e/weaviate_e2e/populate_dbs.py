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

import weaviate.classes as wvc
from neo4j import Driver, GraphDatabase
from weaviate.client import WeaviateClient
from weaviate.collections.classes.types import WeaviateField
from weaviate.connect.helpers import connect_to_local

from ..utils import build_data_objects, populate_neo4j


def populate_dbs(
    neo4j_driver: Driver, w_client: WeaviateClient, collection_name: str = "Jeopardy"
) -> None:
    neo4j_objects, w_question_objs = build_data_objects("weaviate")
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


def populate_weaviate(
    w_client: WeaviateClient,
    w_question_objs: list[wvc.data.DataObject[dict[str, WeaviateField]]],
    collection_name: str,
) -> None:
    questions = w_client.collections.get(collection_name)
    questions.data.insert_many(w_question_objs)


if __name__ == "__main__":
    neo4j_auth = ("neo4j", "password")
    neo4j_url = "neo4j://localhost:7687"
    with GraphDatabase.driver(neo4j_url, auth=neo4j_auth) as neo4j_driver:
        with connect_to_local() as w_client:
            populate_dbs(neo4j_driver, w_client)
