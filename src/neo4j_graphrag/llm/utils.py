import warnings
from typing import Union, Optional

from neo4j_graphrag.message_history import MessageHistory, InMemoryMessageHistory
from neo4j_graphrag.types import LLMMessage


def legacy_inputs_to_message_history(
    input: Union[str, list[LLMMessage], MessageHistory],
    message_history: Optional[Union[list[LLMMessage], MessageHistory]] = None,
    system_instruction: Optional[str] = None,
) -> MessageHistory:
    if message_history:
        warnings.warn(
            "Using message_history parameter is deprecated and will be removed in 2.0. Use a list of inputs or a MessageHistory instead.",
            DeprecationWarning,
        )
        if isinstance(message_history, MessageHistory):
            history = message_history
        else:  # list[LLMMessage]
            history = InMemoryMessageHistory(message_history)
    else:
        history = InMemoryMessageHistory()
    if system_instruction is not None:
        warnings.warn(
            "Using system_instruction parameter is deprecated and will be removed in 2.0. Use a list of inputs or a MessageHistory instead.",
            DeprecationWarning,
        )
        if history.is_empty():
            history.add_message(
                LLMMessage(
                    role="system",
                    content=system_instruction,
                ),
            )
        else:
            warnings.warn(
                "system_instruction provided but ignored as the message history is not empty",
                RuntimeWarning,
            )
    if isinstance(input, str):
        history.add_message(LLMMessage(role="user", content=input))
        return history
    if isinstance(input, list):
        history.add_messages(input)
        return history
    # input is a MessageHistory instance
    history.add_messages(input.messages)
    return history
