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
from unittest import mock
from unittest.mock import MagicMock, call

import pytest
from neo4j_graphrag.exceptions import RagInitializationError, SearchValidationError
from neo4j_graphrag.generation.graphrag import GraphRAG
from neo4j_graphrag.generation.prompts import RagTemplate
from neo4j_graphrag.generation.types import RagResultModel
from neo4j_graphrag.llm import LLMResponse
from neo4j_graphrag.message_history import InMemoryMessageHistory
from neo4j_graphrag.types import LLMMessage, RetrieverResult, RetrieverResultItem


def test_graphrag_prompt_template() -> None:
    template = RagTemplate()
    prompt = template.format(
        context="my context", query_text="user's query", examples=""
    )
    assert (
        prompt
        == """Context:
my context

Examples:


Question:
user's query

Answer:
"""
    )


def test_graphrag_happy_path(retriever_mock: MagicMock, llm: MagicMock) -> None:
    rag = GraphRAG(
        retriever=retriever_mock,
        llm=llm,
    )
    retriever_mock.search.return_value = RetrieverResult(
        items=[
            RetrieverResultItem(content="item content 1"),
            RetrieverResultItem(content="item content 2"),
        ]
    )
    llm.invoke.return_value = LLMResponse(content="llm generated text")

    res = rag.search("question")

    retriever_mock.search.assert_called_once_with(query_text="question")
    llm.invoke.assert_called_once_with(
        """Context:
item content 1
item content 2

Examples:


Question:
question

Answer:
""",
        None,  # message history
        system_instruction="Answer the user question using the provided context.",
    )

    assert isinstance(res, RagResultModel)
    assert res.answer == "llm generated text"
    assert res.retriever_result is None


def test_graphrag_happy_path_with_message_history(
    retriever_mock: MagicMock, llm: MagicMock
) -> None:
    rag = GraphRAG(
        retriever=retriever_mock,
        llm=llm,
    )
    retriever_mock.search.return_value = RetrieverResult(
        items=[
            RetrieverResultItem(content="item content 1"),
            RetrieverResultItem(content="item content 2"),
        ]
    )
    llm.invoke.side_effect = [
        LLMResponse(content="llm generated summary"),
        LLMResponse(content="llm generated text"),
    ]
    message_history = [
        {"role": "user", "content": "initial question"},
        {"role": "assistant", "content": "answer to initial question"},
    ]
    res = rag.search("question", message_history)  # type: ignore

    expected_retriever_query_text = """
Message Summary:
llm generated summary

Current Query:
question
"""

    first_invocation_input = """
Summarize the message history:

user: initial question
assistant: answer to initial question
"""
    first_invocation_system_instruction = "You are a summarization assistant. Summarize the given text in no more than 300 words."
    second_invocation = """Context:
item content 1
item content 2

Examples:


Question:
question

Answer:
"""

    retriever_mock.search.assert_called_once_with(
        query_text=expected_retriever_query_text
    )
    assert llm.invoke.call_count == 2
    llm.invoke.assert_has_calls(
        [
            call(
                input=first_invocation_input,
                system_instruction=first_invocation_system_instruction,
            ),
            call(
                second_invocation,
                message_history,
                system_instruction="Answer the user question using the provided context.",
            ),
        ]
    )

    assert isinstance(res, RagResultModel)
    assert res.answer == "llm generated text"
    assert res.retriever_result is None


def test_graphrag_happy_path_with_in_memory_message_history(
    retriever_mock: MagicMock, llm: MagicMock
) -> None:
    rag = GraphRAG(
        retriever=retriever_mock,
        llm=llm,
    )
    retriever_mock.search.return_value = RetrieverResult(
        items=[
            RetrieverResultItem(content="item content 1"),
            RetrieverResultItem(content="item content 2"),
        ]
    )
    llm.invoke.side_effect = [
        LLMResponse(content="llm generated summary"),
        LLMResponse(content="llm generated text"),
    ]
    message_history = InMemoryMessageHistory(
        messages=[
            LLMMessage(role="user", content="initial question"),
            LLMMessage(role="assistant", content="answer to initial question"),
        ]
    )
    res = rag.search("question", message_history)

    expected_retriever_query_text = """
Message Summary:
llm generated summary

Current Query:
question
"""

    first_invocation_input = """
Summarize the message history:

user: initial question
assistant: answer to initial question
"""
    first_invocation_system_instruction = "You are a summarization assistant. Summarize the given text in no more than 300 words."
    second_invocation = """Context:
item content 1
item content 2

Examples:


Question:
question

Answer:
"""

    retriever_mock.search.assert_called_once_with(
        query_text=expected_retriever_query_text
    )
    assert llm.invoke.call_count == 2
    llm.invoke.assert_has_calls(
        [
            call(
                input=first_invocation_input,
                system_instruction=first_invocation_system_instruction,
            ),
            call(
                second_invocation,
                message_history.messages,
                system_instruction="Answer the user question using the provided context.",
            ),
        ]
    )

    assert isinstance(res, RagResultModel)
    assert res.answer == "llm generated text"
    assert res.retriever_result is None


def test_graphrag_happy_path_custom_system_instruction(
    retriever_mock: MagicMock, llm: MagicMock
) -> None:
    prompt_template = RagTemplate(system_instructions="Custom instruction")
    rag = GraphRAG(
        retriever=retriever_mock,
        llm=llm,
        prompt_template=prompt_template,
    )
    retriever_mock.search.return_value = RetrieverResult(items=[])
    llm.invoke.side_effect = [
        LLMResponse(content="llm generated text"),
    ]
    res = rag.search("question")

    assert llm.invoke.call_count == 1
    llm.invoke.assert_has_calls(
        [
            call(
                mock.ANY,
                None,  # no message history
                system_instruction="Custom instruction",
            ),
        ]
    )

    assert res.answer == "llm generated text"


def test_graphrag_initialization_error(llm: MagicMock) -> None:
    with pytest.raises(RagInitializationError) as excinfo:
        GraphRAG(
            retriever="not a retriever object",  # type: ignore
            llm=llm,
        )
    assert "retriever" in str(excinfo)


def test_graphrag_search_error(retriever_mock: MagicMock, llm: MagicMock) -> None:
    rag = GraphRAG(
        retriever=retriever_mock,
        llm=llm,
    )
    with pytest.raises(SearchValidationError) as excinfo:
        rag.search(10)  # type: ignore
    assert "Input should be a valid string" in str(excinfo)


def test_chat_summary_template(retriever_mock: MagicMock, llm: MagicMock) -> None:
    message_history = [
        {"role": "user", "content": "initial question"},
        {"role": "assistant", "content": "answer to initial question"},
        {"role": "user", "content": "second question"},
        {"role": "assistant", "content": "answer to second question"},
    ]
    rag = GraphRAG(
        retriever=retriever_mock,
        llm=llm,
    )
    prompt = rag._chat_summary_prompt(message_history=message_history)  # type: ignore
    assert (
        prompt
        == """
Summarize the message history:

user: initial question
assistant: answer to initial question
user: second question
assistant: answer to second question
"""
    )


def test_conversation_template(retriever_mock: MagicMock, llm: MagicMock) -> None:
    rag = GraphRAG(
        retriever=retriever_mock,
        llm=llm,
    )
    prompt = rag.conversation_prompt(
        summary="llm generated chat summary", current_query="latest question"
    )
    assert (
        prompt
        == """
Message Summary:
llm generated chat summary

Current Query:
latest question
"""
    )
