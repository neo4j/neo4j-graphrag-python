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

from langchain_community.embeddings import HuggingFaceEmbeddings
from neo4j import GraphDatabase
from neo4j_genai.retrievers.external.pinecone import PineconeNeo4jRetriever
from pinecone import Pinecone

NEO4J_AUTH = ("neo4j", "password")
NEO4J_URL = "neo4j://localhost:7687"
PC_API_KEY = "API_KEY"


def main():
    with GraphDatabase.driver(NEO4J_URL, auth=NEO4J_AUTH) as neo4j_driver:
        pc_client = Pinecone(PC_API_KEY)
        embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        retriever = PineconeNeo4jRetriever(
            driver=neo4j_driver,
            client=pc_client,
            index_name="jeopardy",
            id_property_neo4j="id",
            embedder=embedder,
        )

        res = retriever.search(query_text="biology", top_k=2)
        print(res)


if __name__ == "__main__":
    main()
