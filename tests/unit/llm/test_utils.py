import pytest

from neo4j_graphrag.llm.utils import system_instruction_from_messages, \
    legacy_inputs_to_messages
from neo4j_graphrag.message_history import InMemoryMessageHistory
from neo4j_graphrag.types import LLMMessage


def test_system_instruction_from_messages():
    messages = [
        LLMMessage(role="system", content="text"),
    ]
    assert system_instruction_from_messages(messages) == "text"

    messages = []
    assert system_instruction_from_messages(messages) is None

    messages = [
        LLMMessage(role="assistant", content="text"),
    ]
    assert system_instruction_from_messages(messages) is None


def test_legacy_inputs_to_messages_only_input_as_llm_message_list():
    messages = legacy_inputs_to_messages(input=[
        LLMMessage(role="user", content="text"),
    ])
    assert messages == [
        LLMMessage(role="user", content="text"),
    ]


def test_legacy_inputs_to_messages_only_input_as_message_history():
    messages = legacy_inputs_to_messages(input=InMemoryMessageHistory(
        messages=[
            LLMMessage(role="user", content="text"),
        ]
    ))
    assert messages == [
        LLMMessage(role="user", content="text"),
    ]


def test_legacy_inputs_to_messages_only_input_as_str():
    messages = legacy_inputs_to_messages(input="text")
    assert messages == [
        LLMMessage(role="user", content="text"),
    ]


def test_legacy_inputs_to_messages_input_as_str_and_message_history_as_llm_message_list():
    messages = legacy_inputs_to_messages(
        input="text",
        message_history=[
            LLMMessage(role="assistant", content="How can I assist you today?"),
        ]
    )
    assert messages == [
        LLMMessage(role="assistant", content="How can I assist you today?"),
        LLMMessage(role="user", content="text"),
    ]


def test_legacy_inputs_to_messages_input_as_str_and_message_history_as_message_history():
    messages = legacy_inputs_to_messages(
        input="text",
        message_history=InMemoryMessageHistory(messages=[
            LLMMessage(role="assistant", content="How can I assist you today?"),
        ])
    )
    assert messages == [
        LLMMessage(role="assistant", content="How can I assist you today?"),
        LLMMessage(role="user", content="text"),
    ]


def test_legacy_inputs_to_messages_with_explicit_system_instruction():
    messages = legacy_inputs_to_messages(
        input="text",
        message_history=[
            LLMMessage(role="assistant", content="How can I assist you today?"),
        ],
        system_instruction="You are a genius."
    )
    assert messages == [
        LLMMessage(role="system", content="You are a genius."),
        LLMMessage(role="assistant", content="How can I assist you today?"),
        LLMMessage(role="user", content="text"),
    ]


def test_legacy_inputs_to_messages_do_not_duplicate_system_instruction():
    with pytest.warns(
        UserWarning,
        match="system_instruction provided but ignored as the message history already contains a system message"
    ):
        messages = legacy_inputs_to_messages(
            input="text",
            message_history=[
                LLMMessage(role="system", content="You are super smart."),
            ],
            system_instruction="You are a genius."
        )
        assert messages == [
            LLMMessage(role="system", content="You are super smart."),
            LLMMessage(role="user", content="text"),
        ]
