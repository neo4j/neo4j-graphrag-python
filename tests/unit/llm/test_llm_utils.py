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

from neo4j_graphrag.llm.utils import (
    legacy_inputs_to_messages,
    system_instruction_from_messages,
)
from neo4j_graphrag.message_history import InMemoryMessageHistory
from neo4j_graphrag.types import LLMMessage


def test_system_instruction_from_messages_found() -> None:
    # Covers line 29: return message["content"] when system role present
    messages = [
        LLMMessage(role="system", content="You are helpful"),
        LLMMessage(role="user", content="hi"),
    ]
    assert system_instruction_from_messages(messages) == "You are helpful"


def test_system_instruction_from_messages_not_found() -> None:
    messages = [LLMMessage(role="user", content="hi")]
    assert system_instruction_from_messages(messages) is None


def test_legacy_inputs_with_message_history_instance() -> None:
    # Covers line 44: messages = message_history.messages when MessageHistory passed
    history = InMemoryMessageHistory()
    history.add_message(LLMMessage(role="user", content="previous"))
    result = legacy_inputs_to_messages("follow-up", message_history=history)
    assert result[-1]["content"] == "follow-up"
    assert result[0]["content"] == "previous"


def test_legacy_inputs_system_instruction_conflict_warns() -> None:
    # Covers line 51: warnings.warn when system already in history
    messages = [LLMMessage(role="system", content="existing")]
    with pytest.warns(UserWarning, match="system_instruction provided but ignored"):
        legacy_inputs_to_messages(
            "hi", message_history=messages, system_instruction="new"
        )


def test_legacy_inputs_prompt_as_list() -> None:
    # Covers lines 67-69: isinstance(prompt, list) branch
    prompt_list = [LLMMessage(role="user", content="hello")]
    result = legacy_inputs_to_messages(prompt_list)
    assert result == prompt_list


def test_legacy_inputs_prompt_as_message_history() -> None:
    # Covers lines 70-72: prompt is a MessageHistory instance
    history = InMemoryMessageHistory()
    history.add_message(LLMMessage(role="user", content="from history"))
    result = legacy_inputs_to_messages(history)
    assert len(result) == 1
    assert result[0]["content"] == "from history"
