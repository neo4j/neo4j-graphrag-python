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

from typing import Any, List, Optional, Union, cast, Sequence

from pydantic import ValidationError

from neo4j_graphrag.exceptions import LLMGenerationError
from neo4j_graphrag.llm.base import LLMInterface
from neo4j_graphrag.llm.types import (
    BaseMessage,
    LLMResponse,
    MessageList,
    ToolCall,
    ToolCallResponse,
)
from neo4j_graphrag.message_history import MessageHistory
from neo4j_graphrag.tool import Tool
from neo4j_graphrag.types import LLMMessage

try:
    from vertexai.generative_models import (
        Content,
        FunctionCall,
        FunctionDeclaration,
        GenerationResponse,
        GenerativeModel,
        Part,
        ResponseValidationError,
        Tool as VertexAITool,
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
        self,
        input: str,
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
    ) -> list[Content]:
        messages = []
        if message_history:
            if isinstance(message_history, MessageHistory):
                message_history = message_history.messages
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
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
    ) -> LLMResponse:
        """Sends text to the LLM and returns a response.

        Args:
            input (str): The text to send to the LLM.
            message_history (Optional[Union[List[LLMMessage], MessageHistory]]): A collection previous messages,
                with each message having a specific role assigned.
            system_instruction (Optional[str]): An option to override the llm system message for this invocation.

        Returns:
            LLMResponse: The response from the LLM.
        """
        model = self._get_model(
            system_instruction=system_instruction,
            tools=None,
        )
        try:
            if isinstance(message_history, MessageHistory):
                message_history = message_history.messages
            messages = self.get_messages(input, message_history)
            response = model.generate_content(messages)
            return self._parse_content_response(response)
        except ResponseValidationError as e:
            raise LLMGenerationError("Error calling LLM") from e

    async def ainvoke(
        self,
        input: str,
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
    ) -> LLMResponse:
        """Asynchronously sends text to the LLM and returns a response.

        Args:
            input (str): The text to send to the LLM.
            message_history (Optional[Union[List[LLMMessage], MessageHistory]]): A collection previous messages,
                with each message having a specific role assigned.
            system_instruction (Optional[str]): An option to override the llm system message for this invocation.

        Returns:
            LLMResponse: The response from the LLM.
        """
        try:
            if isinstance(message_history, MessageHistory):
                message_history = message_history.messages
            model = self._get_model(
                system_instruction=system_instruction,
                tools=None,
            )
            messages = self.get_messages(input, message_history)
            response = await model.generate_content_async(messages)
            return self._parse_content_response(response)
        except ResponseValidationError as e:
            raise LLMGenerationError(e)

    def _to_vertexai_function_declaration(self, tool: Tool) -> FunctionDeclaration:
        return FunctionDeclaration(
            name=tool.get_name(),
            description=tool.get_description(),
            parameters=tool.get_parameters(exclude=["additional_properties"]),
        )

    def _get_llm_tools(
        self, tools: Optional[Sequence[Tool]]
    ) -> Optional[list[VertexAITool]]:
        if not tools:
            return None
        return [
            VertexAITool(
                function_declarations=[
                    self._to_vertexai_function_declaration(tool) for tool in tools
                ]
            )
        ]

    def _get_model(
        self,
        system_instruction: Optional[str] = None,
        tools: Optional[Sequence[Tool]] = None,
    ) -> GenerativeModel:
        system_message = [system_instruction] if system_instruction is not None else []
        vertex_ai_tools = self._get_llm_tools(tools)
        options = dict(self.options)
        if tools:
            # we want a tool back, remove generation_config if defined
            options.pop("generation_config", None)
        else:
            # no tools, remove tool_config if defined
            options.pop("tool_config", None)
        model = GenerativeModel(
            model_name=self.model_name,
            system_instruction=system_message,
            tools=vertex_ai_tools,
            **options,
        )
        return model

    async def _acall_llm(
        self,
        input: str,
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
        tools: Optional[Sequence[Tool]] = None,
    ) -> GenerationResponse:
        model = self._get_model(system_instruction=system_instruction, tools=tools)
        messages = self.get_messages(input, message_history)
        response = await model.generate_content_async(messages, **self.model_params)
        return response

    def _call_llm(
        self,
        input: str,
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
        tools: Optional[Sequence[Tool]] = None,
    ) -> GenerationResponse:
        model = self._get_model(system_instruction=system_instruction, tools=tools)
        messages = self.get_messages(input, message_history)
        response = model.generate_content(messages, **self.model_params)
        return response

    def _to_tool_call(self, function_call: FunctionCall) -> ToolCall:
        return ToolCall(
            name=function_call.name,
            arguments=function_call.args,
        )

    def _parse_tool_response(self, response: GenerationResponse) -> ToolCallResponse:
        function_calls = response.candidates[0].function_calls
        return ToolCallResponse(
            tool_calls=[self._to_tool_call(f) for f in function_calls],
            content=None,
        )

    def _parse_content_response(self, response: GenerationResponse) -> LLMResponse:
        return LLMResponse(
            content=response.text,
        )

    async def ainvoke_with_tools(
        self,
        input: str,
        tools: Sequence[Tool],
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
    ) -> ToolCallResponse:
        response = await self._acall_llm(
            input,
            message_history=message_history,
            system_instruction=system_instruction,
            tools=tools,
        )
        return self._parse_tool_response(response)

    def invoke_with_tools(
        self,
        input: str,
        tools: Sequence[Tool],
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
    ) -> ToolCallResponse:
        response = self._call_llm(
            input,
            message_history=message_history,
            system_instruction=system_instruction,
            tools=tools,
        )
        return self._parse_tool_response(response)
