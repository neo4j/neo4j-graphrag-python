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
import neo4j
from neo4j_graphrag.llm.types import LLMMessage
from neo4j_graphrag.message_history import Neo4jMessageHistory


def test_neo4j_message_history_add_message(driver: neo4j.Driver) -> None:
    driver.execute_query(query_="MATCH (n) DETACH DELETE n;")
    message_history = Neo4jMessageHistory(session_id="123", driver=driver)
    message_history.add_message(
        LLMMessage(role="user", content="Hello"),
    )
    assert len(message_history.messages) == 1
    assert message_history.messages[0]["role"] == "user"
    assert message_history.messages[0]["content"] == "Hello"


def test_neo4j_message_history_add_messages(driver: neo4j.Driver) -> None:
    driver.execute_query(query_="MATCH (n) DETACH DELETE n;")
    message_history = Neo4jMessageHistory(session_id="123", driver=driver)
    message_history.add_messages(
        [
            LLMMessage(role="system", content="You are a helpful assistant."),
            LLMMessage(role="user", content="Hello"),
            LLMMessage(
                role="assistant",
                content="Hello, how may I help you today?",
            ),
            LLMMessage(role="user", content="I'd like to buy a new car."),
            LLMMessage(
                role="assistant",
                content="I'd be happy to help you find the perfect car.",
            ),
        ]
    )
    assert len(message_history.messages) == 5
    assert message_history.messages[0]["role"] == "system"
    assert message_history.messages[0]["content"] == "You are a helpful assistant."
    assert message_history.messages[1]["role"] == "user"
    assert message_history.messages[1]["content"] == "Hello"
    assert message_history.messages[2]["role"] == "assistant"
    assert message_history.messages[2]["content"] == "Hello, how may I help you today?"
    assert message_history.messages[3]["role"] == "user"
    assert message_history.messages[3]["content"] == "I'd like to buy a new car."
    assert message_history.messages[4]["role"] == "assistant"
    assert (
        message_history.messages[4]["content"]
        == "I'd be happy to help you find the perfect car."
    )


def test_neo4j_message_history_clear(driver: neo4j.Driver) -> None:
    driver.execute_query(query_="MATCH (n) DETACH DELETE n;")
    message_history = Neo4jMessageHistory(session_id="123", driver=driver)
    message_history.add_messages(
        [
            LLMMessage(role="system", content="You are a helpful assistant."),
            LLMMessage(role="user", content="Hello"),
        ]
    )
    assert len(message_history.messages) == 2
    message_history.clear()
    assert len(message_history.messages) == 0


def test_neo4j_message_window_size(driver: neo4j.Driver) -> None:
    driver.execute_query(query_="MATCH (n) DETACH DELETE n;")
    message_history = Neo4jMessageHistory(session_id="123", driver=driver, window=1)
    message_history.add_messages(
        [
            LLMMessage(role="system", content="You are a helpful assistant."),
            LLMMessage(role="user", content="Hello"),
            LLMMessage(
                role="assistant",
                content="Hello, how may I help you today?",
            ),
            LLMMessage(role="user", content="I'd like to buy a new car."),
            LLMMessage(
                role="assistant",
                content="I'd be happy to help you find the perfect car.",
            ),
        ]
    )
    assert len(message_history.messages) == 1
    assert (
        message_history.messages[0]["content"]
        == "I'd be happy to help you find the perfect car."
    )
    assert message_history.messages[0]["role"] == "assistant"
