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

import pytest

from neo4j_genai import VectorRetriever
from neo4j_genai.generation.rag import RAG


@pytest.mark.usefixtures("setup_neo4j_for_retrieval")
def test_rag_happy_path(driver, custom_embedder, llm):
    retriever = VectorRetriever(driver, "vector-index-name", custom_embedder)
    rag = RAG(
        retriever=retriever,
        llm=llm,
    )
    rag.llm.invoke.return_value = "some text"

    result = rag.search(
        query="Find me a book about Fremen",
        retriever_config={
            "top_k": 5,
        },
    )

    llm.invoke.assert_called_once()
    assert isinstance(result, str)
    assert result == "some text"
