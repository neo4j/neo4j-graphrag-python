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

from typing import Any, Optional, Sequence


from neo4j_graphrag.exceptions import LLMGenerationError
from neo4j_graphrag.llm.base import LLMInterface
from neo4j_graphrag.utils.rate_limit import (
    RateLimitHandler,
)
from neo4j_graphrag.llm.types import (
    LLMResponse,
    ToolCall,
    ToolCallResponse,
)
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
        ToolConfig,
    )
except ImportError:
    GenerativeModel = None  # type: ignore[misc, assignment]
    ResponseValidationError = None  # type: ignore[misc, assignment]


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
        rate_limit_handler: Optional[RateLimitHandler] = None,
        **kwargs: Any,
    ):
        if GenerativeModel is None or ResponseValidationError is None:
            raise ImportError(
                """Could not import Vertex AI Python client.
                Please install it with `pip install "neo4j-graphrag[google]"`."""
            )
        super().__init__(model_name, model_params, rate_limit_handler)
        self.model_name = model_name
        self.system_instruction = system_instruction
        self.options = kwargs

    def get_messages(
        self,
        input: list[LLMMessage],
    ) -> tuple[str | None, list[Content]]:
        messages = []
        system_instruction = self.system_instruction
        for message in input:
            if message.get("role") == "system":
                system_instruction = message.get("content")
                continue
            if message.get("role") == "user":
                messages.append(
                    Content(
                        role="user",
                        parts=[Part.from_text(message.get("content", ""))],
                    )
                )
                continue
            if message.get("role") == "assistant":
                messages.append(
                    Content(
                        role="model",
                        parts=[Part.from_text(message.get("content", ""))],
                    )
                )
                continue
        return system_instruction, messages

    def _invoke(
        self,
        input: list[LLMMessage],
    ) -> LLMResponse:
        """Sends text to the LLM and returns a response.

        Args:
            input (str): The text to send to the LLM.

        Returns:
            LLMResponse: The response from the LLM.
        """
        system_instruction, messages = self.get_messages(input)
        model = self._get_model(
            system_instruction=system_instruction,
        )
        try:
            options = self._get_call_params(messages, tools=None)
            response = model.generate_content(**options)
            return self._parse_content_response(response)
        except ResponseValidationError as e:
            raise LLMGenerationError("Error calling VertexAILLM") from e

    async def _ainvoke(
        self,
        input: list[LLMMessage],
    ) -> LLMResponse:
        """Asynchronously sends text to the LLM and returns a response.

        Args:
            input (str): The text to send to the LLM.

        Returns:
            LLMResponse: The response from the LLM.
        """
        try:
            system_instruction, messages = self.get_messages(input)
            model = self._get_model(
                system_instruction=system_instruction,
            )
            options = self._get_call_params(messages, tools=None)
            response = await model.generate_content_async(**options)
            return self._parse_content_response(response)
        except ResponseValidationError as e:
            raise LLMGenerationError("Error calling VertexAILLM") from e

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
    ) -> GenerativeModel:
        model = GenerativeModel(
            model_name=self.model_name,
            system_instruction=system_instruction,
        )
        return model

    def _get_call_params(
        self,
        contents: list[Content],
        tools: Optional[Sequence[Tool]],
    ) -> dict[str, Any]:
        options = dict(self.options)
        if tools:
            # we want a tool back, remove generation_config if defined
            options.pop("generation_config", None)
            options["tools"] = self._get_llm_tools(tools)
            if "tool_config" not in options:
                options["tool_config"] = ToolConfig(
                    function_calling_config=ToolConfig.FunctionCallingConfig(
                        mode=ToolConfig.FunctionCallingConfig.Mode.ANY,
                    )
                )
        else:
            # no tools, remove tool_config if defined
            options.pop("tool_config", None)
        options["contents"] = contents
        return options

    async def _acall_llm(
        self,
        input: list[LLMMessage],
        tools: Optional[Sequence[Tool]] = None,
    ) -> GenerationResponse:
        system_instruction, contents = self.get_messages(input)
        model = self._get_model(system_instruction)
        options = self._get_call_params(contents, tools)
        response = await model.generate_content_async(**options)
        return response  # type: ignore[no-any-return]

    def _call_llm(
        self,
        input: list[LLMMessage],
        tools: Optional[Sequence[Tool]] = None,
    ) -> GenerationResponse:
        system_instruction, contents = self.get_messages(input)
        model = self._get_model(system_instruction)
        options = self._get_call_params(contents, tools)
        response = model.generate_content(**options)
        return response  # type: ignore[no-any-return]

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

    async def _ainvoke_with_tools(
        self,
        input: list[LLMMessage],
        tools: Sequence[Tool],
    ) -> ToolCallResponse:
        response = await self._acall_llm(
            input,
            tools=tools,
        )
        return self._parse_tool_response(response)

    def _invoke_with_tools(
        self,
        input: list[LLMMessage],
        tools: Sequence[Tool],
    ) -> ToolCallResponse:
        response = self._call_llm(
            input,
            tools=tools,
        )
        return self._parse_tool_response(response)
