from typing import Type, Generator
from unittest.mock import patch, Mock

import pytest
from joblib.testing import fixture
from pydantic import ValidationError

from neo4j_graphrag.exceptions import LLMGenerationError
from neo4j_graphrag.llm import LLMInterface
from neo4j_graphrag.types import LLMMessage


@fixture(scope="module")  # type: ignore[misc]
def llm_interface() -> Generator[Type[LLMInterface], None, None]:
    real_abstract_methods = LLMInterface.__abstractmethods__
    LLMInterface.__abstractmethods__ = frozenset()

    class CustomLLMInterface(LLMInterface):
        pass

    yield CustomLLMInterface

    LLMInterface.__abstractmethods__ = real_abstract_methods


@patch("neo4j_graphrag.llm.base.legacy_inputs_to_messages")
def test_base_llm_interface_invoke_with_input_as_str(
    mock_inputs: Mock, llm_interface: Type[LLMInterface]
) -> None:
    mock_inputs.return_value = [
        LLMMessage(
            role="user",
            content="return value of the legacy_inputs_to_messages function",
        )
    ]
    llm = llm_interface(model_name="test")
    message_history = [
        LLMMessage(
            **{"role": "user", "content": "When does the sun come up in the summer?"}
        ),
        LLMMessage(**{"role": "assistant", "content": "Usually around 6am."}),
    ]
    question = "What about next season?"
    system_instruction = "You are a genius."

    with patch.object(llm, "_invoke") as mock_invoke:
        llm.invoke(question, message_history, system_instruction)
        mock_invoke.assert_called_once_with(
            [
                LLMMessage(
                    role="user",
                    content="return value of the legacy_inputs_to_messages function",
                )
            ]
        )
    mock_inputs.assert_called_once_with(
        question,
        message_history,
        system_instruction,
    )


@patch("neo4j_graphrag.llm.base.legacy_inputs_to_messages")
def test_base_llm_interface_invoke_with_invalid_inputs(
    mock_inputs: Mock, llm_interface: Type[LLMInterface]
) -> None:
    mock_inputs.side_effect = [
        ValidationError.from_exception_data("Invalid data", line_errors=[])
    ]
    llm = llm_interface(model_name="test")
    question = "What about next season?"

    with pytest.raises(LLMGenerationError, match="Input validation failed"):
        llm.invoke(question)
    mock_inputs.assert_called_once_with(
        question,
        None,
        None,
    )


@patch("neo4j_graphrag.llm.base.legacy_inputs_to_messages")
def test_base_llm_interface_invoke_with_tools_with_input_as_str(
    mock_inputs: Mock, llm_interface: Type[LLMInterface]
) -> None:
    mock_inputs.return_value = [
        LLMMessage(
            role="user",
            content="return value of the legacy_inputs_to_messages function",
        )
    ]
    llm = llm_interface(model_name="test")
    message_history = [
        LLMMessage(
            **{"role": "user", "content": "When does the sun come up in the summer?"}
        ),
        LLMMessage(**{"role": "assistant", "content": "Usually around 6am."}),
    ]
    question = "What about next season?"
    system_instruction = "You are a genius."

    with patch.object(llm, "_invoke_with_tools") as mock_invoke:
        llm.invoke_with_tools(question, [], message_history, system_instruction)
        mock_invoke.assert_called_once_with(
            [
                LLMMessage(
                    role="user",
                    content="return value of the legacy_inputs_to_messages function",
                )
            ],
            [],  # tools
        )
    mock_inputs.assert_called_once_with(
        question,
        message_history,
        system_instruction,
    )


@patch("neo4j_graphrag.llm.base.legacy_inputs_to_messages")
def test_base_llm_interface_invoke_with_tools_with_invalid_inputs(
    mock_inputs: Mock, llm_interface: Type[LLMInterface]
) -> None:
    mock_inputs.side_effect = [
        ValidationError.from_exception_data("Invalid data", line_errors=[])
    ]
    llm = llm_interface(model_name="test")
    question = "What about next season?"

    with pytest.raises(LLMGenerationError, match="Input validation failed"):
        llm.invoke_with_tools(question, [])
    mock_inputs.assert_called_once_with(
        question,
        None,
        None,
    )
