from typing import Type, Generator, Optional, Any
from unittest.mock import patch, Mock

from joblib.testing import fixture

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
def test_base_llm_interface_invoke_with_input_as_str(mock_inputs: Mock, llm_interface: Type[LLMInterface]) -> None:
    mock_inputs.return_value = [LLMMessage(role="user", content="return value of the legacy_inputs_to_messages function")]
    llm = llm_interface(model_name="test")
    message_history = [
        LLMMessage(**{"role": "user", "content": "When does the sun come up in the summer?"}),
        LLMMessage(**{"role": "assistant", "content": "Usually around 6am."}),
    ]
    question = "What about next season?"
    system_instruction = "You are a genius."

    with patch.object(llm, "_invoke") as mock_invoke:
        llm.invoke(question, message_history, system_instruction)
        mock_invoke.assert_called_once_with(
            [LLMMessage(role="user", content="return value of the legacy_inputs_to_messages function")]
        )
    mock_inputs.assert_called_once_with(
        question,
        message_history,
        system_instruction,
    )
