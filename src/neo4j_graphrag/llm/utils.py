import warnings
from typing import Union, Optional

from neo4j_graphrag.message_history import MessageHistory
from neo4j_graphrag.types import LLMMessage


def system_instruction_from_messages(messages: list[LLMMessage]) -> str | None:
    for message in messages:
        if message["role"] == "system":
            return message["content"]
    return None


def legacy_inputs_to_messages(
    input: Union[str, list[LLMMessage], MessageHistory],
    message_history: Optional[Union[list[LLMMessage], MessageHistory]] = None,
    system_instruction: Optional[str] = None,
) -> list[LLMMessage]:
    if message_history:
        warnings.warn(
            "Using message_history parameter is deprecated and will be removed in 2.0. Use a list of inputs or a MessageHistory instead.",
            DeprecationWarning,
        )
        if isinstance(message_history, MessageHistory):
            messages = message_history.messages
        else:  # list[LLMMessage]
            messages = []
    else:
        messages = []
    if system_instruction is not None:
        warnings.warn(
            "Using system_instruction parameter is deprecated and will be removed in 2.0. Use a list of inputs or a MessageHistory instead.",
            DeprecationWarning,
        )
        if system_instruction_from_messages(messages) is not None:
            warnings.warn(
                "system_instruction provided but ignored as the message history already contains a system message",
                RuntimeWarning,
            )
        else:
            messages.append(
                LLMMessage(
                    role="system",
                    content=system_instruction,
                ),
            )

    if isinstance(input, str):
        messages.append(LLMMessage(role="user", content=input))
        return messages
    if isinstance(input, list):
        messages.extend(input)
        return messages
    # input is a MessageHistory instance
    messages.extend(input.messages)
    return messages
