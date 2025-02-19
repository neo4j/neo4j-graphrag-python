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
from unittest.mock import MagicMock

import pytest
from neo4j_graphrag.llm.types import LLMMessage
from neo4j_graphrag.message_history import InMemoryMessageHistory, Neo4jMessageHistory
from pydantic import ValidationError


def test_in_memory_message_history_add_message() -> None:
    message_history = InMemoryMessageHistory()
    message_history.add_message(
        LLMMessage(role="user", content="may thy knife chip and shatter")
    )
    assert len(message_history.messages) == 1
    assert message_history.messages[0]["role"] == "user"
    assert message_history.messages[0]["content"] == "may thy knife chip and shatter"


def test_in_memory_message_history_add_messages() -> None:
    message_history = InMemoryMessageHistory()
    message_history.add_messages(
        [
            LLMMessage(role="user", content="may thy knife chip and shatter"),
            LLMMessage(
                role="assistant",
                content="He who controls the spice controls the universe.",
            ),
        ]
    )
    assert len(message_history.messages) == 2
    assert message_history.messages[0]["role"] == "user"
    assert message_history.messages[0]["content"] == "may thy knife chip and shatter"
    assert message_history.messages[1]["role"] == "assistant"
    assert (
        message_history.messages[1]["content"]
        == "He who controls the spice controls the universe."
    )


def test_in_memory_message_history_clear() -> None:
    message_history = InMemoryMessageHistory()
    message_history.add_messages(
        [
            LLMMessage(role="user", content="may thy knife chip and shatter"),
            LLMMessage(
                role="assistant",
                content="He who controls the spice controls the universe.",
            ),
        ]
    )
    assert len(message_history.messages) == 2
    message_history.clear()
    assert len(message_history.messages) == 0


def test_neo4j_message_history_invalid_session_id(driver: MagicMock) -> None:
    with pytest.raises(ValidationError) as exc_info:
        Neo4jMessageHistory(session_id=1.5, driver=driver, window=1)  # type: ignore[arg-type]
    assert "Input should be a valid string" in str(exc_info.value)


def test_neo4j_message_history_invalid_driver() -> None:
    with pytest.raises(ValidationError) as exc_info:
        Neo4jMessageHistory(session_id="123", driver=1.5, window=1)  # type: ignore[arg-type]
    assert "Input should be an instance of Driver" in str(exc_info.value)


def test_neo4j_message_history_invalid_window(driver: MagicMock) -> None:
    with pytest.raises(ValidationError) as exc_info:
        Neo4jMessageHistory(session_id="123", driver=driver, window=-1)
    assert "Input should be greater than 0" in str(exc_info.value)


def test_neo4j_message_history_messages_setter(driver: MagicMock) -> None:
    message_history = Neo4jMessageHistory(session_id="123", driver=driver)
    with pytest.raises(NotImplementedError) as exc_info:
        message_history.messages = [
            LLMMessage(role="user", content="may thy knife chip and shatter"),
        ]
    assert (
        str(exc_info.value)
        == "Direct assignment to 'messages' is not allowed. Use the 'add_messages' instead."
    )
