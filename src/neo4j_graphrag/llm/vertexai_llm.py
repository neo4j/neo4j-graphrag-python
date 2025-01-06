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
from __future__ import annotations

from typing import Any, Optional, cast

from pydantic import ValidationError

from neo4j_graphrag.exceptions import LLMGenerationError
from neo4j_graphrag.llm.base import LLMInterface
from neo4j_graphrag.llm.types import BaseMessage, LLMMessage, LLMResponse, MessageList

try:
    from vertexai.generative_models import (
        Content,
        GenerativeModel,
        Part,
        ResponseValidationError,
    )
except ImportError:
    GenerativeModel = None
    ResponseValidationError = None


class VertexAILLM(LLMInterface):
    """Interface for large language models on Vertex AI

    Args:
        model_name (str, optional): Name of the LLM to use. Defaults to "gemini-1.5-flash-001".
        model_params (Optional[dict], optional): Additional parameters passed to the model when text is sent to it. Defaults to None.
        system_instruction: Optional[str], optional): Additional instructions for setting the behavior and context for the model in a conversation. Defaults to None.
        **kwargs (Any): Arguments passed to the model when for the class is initialised. Defaults to None.

    Raises:
        LLMGenerationError: If there's an error generating the response from the model.

    Example:

    .. code-block:: python

        from neo4j_graphrag.llm import VertexAILLM
        from vertexai.generative_models import GenerationConfig

        generation_config = GenerationConfig(temperature=0.0)
        llm = VertexAILLM(
            model_name="gemini-1.5-flash-001", generation_config=generation_config
        )
        llm.invoke("Who is the mother of Paul Atreides?")
    """

    def __init__(
        self,
        model_name: str = "gemini-1.5-flash-001",
        model_params: Optional[dict[str, Any]] = None,
        system_instruction: Optional[str] = None,
        **kwargs: Any,
    ):
        if GenerativeModel is None or ResponseValidationError is None:
            raise ImportError(
                """Could not import Vertex AI Python client.
                Please install it with `pip install "neo4j-graphrag[google]"`."""
            )
        super().__init__(model_name, model_params)
        self.model_name = model_name
        self.system_instruction = system_instruction
        self.options = kwargs

    def get_messages(
        self, input: str, message_history: Optional[list[LLMMessage]] = None
    ) -> list[Content]:
        messages = []
        if message_history:
            try:
                MessageList(messages=cast(list[BaseMessage], message_history))
            except ValidationError as e:
                raise LLMGenerationError(e.errors()) from e

            for message in message_history:
                if message.get("role") == "user":
                    messages.append(
                        Content(
                            role="user", parts=[Part.from_text(message.get("content"))]
                        )
                    )
                elif message.get("role") == "assistant":
                    messages.append(
                        Content(
                            role="model", parts=[Part.from_text(message.get("content"))]
                        )
                    )

        messages.append(Content(role="user", parts=[Part.from_text(input)]))
        return messages

    def invoke(
        self,
        input: str,
        message_history: Optional[list[LLMMessage]] = None,
        system_instruction: Optional[str] = None,
    ) -> LLMResponse:
        """Sends text to the LLM and returns a response.

        Args:
            input (str): The text to send to the LLM.
            message_history (Optional[list]): A collection previous messages, with each message having a specific role assigned.
            system_instruction (Optional[str]): An option to override the llm system message for this invokation.

        Returns:
            LLMResponse: The response from the LLM.
        """
        system_instruction = (
            system_instruction
            if system_instruction is not None
            else self.system_instruction
        )
        system_message = [system_instruction] if system_instruction is not None else []
        self.model = GenerativeModel(
            model_name=self.model_name,
            system_instruction=system_message,
            **self.options,
        )
        try:
            messages = self.get_messages(input, message_history)
            response = self.model.generate_content(messages, **self.model_params)
            return LLMResponse(content=response.text)
        except ResponseValidationError as e:
            raise LLMGenerationError(e)

    async def ainvoke(
        self,
        input: str,
        message_history: Optional[list[LLMMessage]] = None,
        system_instruction: Optional[str] = None,
    ) -> LLMResponse:
        """Asynchronously sends text to the LLM and returns a response.

        Args:
            input (str): The text to send to the LLM.
            message_history (Optional[list]): A collection previous messages, with each message having a specific role assigned.
            system_instruction (Optional[str]): An option to override the llm system message for this invokation.

        Returns:
            LLMResponse: The response from the LLM.
        """
        try:
            system_message = (
                system_instruction
                if system_instruction is not None
                else self.system_instruction
            )
            self.model = GenerativeModel(
                model_name=self.model_name,
                system_instruction=[system_message],
                **self.options,
            )
            messages = self.get_messages(input, message_history)
            response = await self.model.generate_content_async(
                messages, **self.model_params
            )
            return LLMResponse(content=response.text)
        except ResponseValidationError as e:
            raise LLMGenerationError(e)
