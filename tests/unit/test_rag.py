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

from neo4j_genai.generation.prompts import RagTemplate
from neo4j_genai.generation.rag import RAG
from neo4j_genai.generation.types import RagResultModel
from neo4j_genai.types import RetrieverResult, RetrieverResultItem


def test_rag_prompt_template():
    template = RagTemplate()
    prompt = template.format(context="my context", query="user's query", examples="")
    assert (
        prompt
        == """Answer the user question using the following context

Context:
my context

Examples:


Question:
user's query

Answer:
"""
    )


def test_rag_happy_path(driver, retriever_mock, llm):
    rag = RAG(
        retriever=retriever_mock,
        llm=llm,
    )
    retriever_mock.search.return_value = RetrieverResult(
        items=[
            RetrieverResultItem(content="item content 1"),
            RetrieverResultItem(content="item content 2"),
        ]
    )
    llm.invoke.return_value = "llm generated text"

    res = rag.search("question")

    retriever_mock.search.assert_called_once_with(query_text="question")
    llm.invoke.assert_called_once_with("""Answer the user question using the following context

Context:
item content 1
item content 2

Examples:


Question:
question

Answer:
""")

    assert isinstance(res, RagResultModel)
    assert res.answer == "llm generated text"
    assert res.retriever_result is None


def test_rag_initialization_error(llm):
    with pytest.raises(ValueError) as excinfo:
        RAG(
            retriever="not a retriever object",
            llm=llm,
        )
    assert "Input should be an instance of Retriever" in str(excinfo)


def test_rag_search_error(driver, retriever_mock, llm):
    rag = RAG(
        retriever=retriever_mock,
        llm=llm,
    )
    with pytest.raises(ValueError) as excinfo:
        rag.search(10)
    assert "Input should be a valid string" in str(excinfo)
