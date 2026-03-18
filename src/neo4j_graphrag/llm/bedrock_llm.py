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

from __future__ import annotations

import asyncio
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Union,
    cast,
    overload,
)

from pydantic import ValidationError

from neo4j_graphrag.exceptions import LLMGenerationError
from neo4j_graphrag.message_history import MessageHistory
from neo4j_graphrag.tool import Tool
from neo4j_graphrag.types import LLMMessage
from neo4j_graphrag.utils.rate_limit import (
    RateLimitHandler,
    async_rate_limit_handler as async_rate_limit_handler_decorator,
    rate_limit_handler as rate_limit_handler_decorator,
)

from .base import LLMInterface, LLMInterfaceV2
from .types import (
    BaseMessage,
    LLMResponse,
    MessageList,
    ToolCall,
    ToolCallResponse,
)

if TYPE_CHECKING:
    from mypy_boto3_bedrock_runtime import BedrockRuntimeClient
    from mypy_boto3_bedrock_runtime.type_defs import (
        MessageTypeDef,
        SystemContentBlockTypeDef,
    )

try:
    import boto3
except ImportError:
    boto3 = None


class BedrockLLM(LLMInterface, LLMInterfaceV2):
    """AWS Bedrock LLM provider using the Converse API.

    This class provides access to foundation models on AWS Bedrock through
    the unified Converse API. It supports Claude 4.x models and other
    Bedrock-hosted models with a consistent interface.

    Note:
        Newer models (Claude Sonnet 4.5, Claude 3.5, etc.) require inference
        profile IDs instead of direct model IDs. The format is
        ``{region}.{provider}.{model}``, e.g., ``us.anthropic.claude-sonnet-4-5-20250929-v1:0``.
        See https://docs.aws.amazon.com/bedrock/latest/userguide/inference-profiles-support.html

    Args:
        model_id (str): The Bedrock model or inference profile identifier.
            Defaults to Claude Sonnet 4.5 (US inference profile).
        region_name (str, optional): AWS region name. Falls back to
            AWS_REGION or AWS_DEFAULT_REGION environment variable.
        inference_profile_id (str, optional): Inference profile ARN for
            cross-region inference. When provided, used instead of model_id.
        client: A pre-configured boto3 bedrock-runtime client.
            If provided, region_name is ignored.
        model_params (dict, optional): Additional parameters passed to the
            Converse API (temperature, maxTokens, topP, etc.).
        rate_limit_handler (RateLimitHandler, optional): Handler for rate limiting.
        **kwargs: Additional arguments passed to boto3.client() if client
            is not provided.

    Example:
        >>> from neo4j_graphrag.llm import BedrockLLM
        >>> llm = BedrockLLM(region_name="us-east-1")
        >>> response = llm.invoke("What is the capital of France?")
        >>> print(response.content)

    Example with inference profile:
        >>> llm = BedrockLLM(
        ...     inference_profile_id="arn:aws:bedrock:us-east-1:123456789:inference-profile/my-profile"
        ... )

    Example with custom model:
        >>> llm = BedrockLLM(
        ...     model_id="us.anthropic.claude-3-5-haiku-20241022-v1:0",
        ...     model_params={"temperature": 0.7, "maxTokens": 1000}
        ... )
    """

    def __init__(
        self,
        model_id: str = "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        region_name: Optional[str] = None,
        inference_profile_id: Optional[str] = None,
        client: Optional[Any] = None,
        model_params: Optional[dict[str, Any]] = None,
        rate_limit_handler: Optional[RateLimitHandler] = None,
        **kwargs: Any,
    ) -> None:
        if boto3 is None:
            raise ImportError(
                """Could not import boto3 python client.
                Please install it with `pip install "neo4j-graphrag[bedrock]"`."""
            )

        LLMInterfaceV2.__init__(
            self,
            model_name=model_id,
            model_params=model_params or {},
            rate_limit_handler=rate_limit_handler,
            **kwargs,
        )

        self.model_id = model_id
        self.inference_profile_id = inference_profile_id

        if client is not None:
            self.client: BedrockRuntimeClient = client
        else:
            self.client = boto3.client(
                "bedrock-runtime",
                region_name=region_name,
                **kwargs,
            )

    def _get_model_identifier(self) -> str:
        """Get the model identifier to use for API calls."""
        return self.inference_profile_id or self.model_id

    # Overloads for LLMInterface and LLMInterfaceV2 methods
    @overload  # type: ignore[no-overload-impl]
    def invoke(
        self,
        input: str,
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
    ) -> LLMResponse: ...

    @overload
    def invoke(
        self,
        input: List[LLMMessage],
        **kwargs: Any,
    ) -> LLMResponse: ...

    @overload  # type: ignore[no-overload-impl]
    async def ainvoke(
        self,
        input: str,
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
    ) -> LLMResponse: ...

    @overload
    async def ainvoke(
        self,
        input: List[LLMMessage],
        **kwargs: Any,
    ) -> LLMResponse: ...

    # Switching logic for LLMInterface or LLMInterfaceV2
    def invoke(  # type: ignore[no-redef]
        self,
        input: Union[str, List[LLMMessage]],
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        if isinstance(input, str):
            return self.__invoke_v1(input, message_history, system_instruction)
        elif isinstance(input, list):
            return self.__invoke_v2(input, **kwargs)
        else:
            raise ValueError(f"Invalid input type for invoke method - {type(input)}")

    async def ainvoke(  # type: ignore[no-redef]
        self,
        input: Union[str, List[LLMMessage]],
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        if isinstance(input, str):
            return await self.__ainvoke_v1(input, message_history, system_instruction)
        elif isinstance(input, list):
            return await self.__ainvoke_v2(input, **kwargs)
        else:
            raise ValueError(f"Invalid input type for ainvoke method - {type(input)}")

    @rate_limit_handler_decorator
    def __invoke_v1(
        self,
        input: str,
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
    ) -> LLMResponse:
        """Sends text to the LLM and returns a response (V1 interface).

        Args:
            input: The text to send to the LLM.
            message_history: A collection of previous messages.
            system_instruction: Optional system message override.

        Returns:
            LLMResponse: The response from the LLM.

        Raises:
            LLMGenerationError: If text generation fails.
        """
        try:
            if isinstance(message_history, MessageHistory):
                message_history = message_history.messages

            messages = self.get_messages(input, message_history)
            system = self.get_system(system_instruction)

            converse_params: Dict[str, Any] = {
                "modelId": self._get_model_identifier(),
                "messages": messages,
            }
            if system:
                converse_params["system"] = system
            if self.model_params:
                converse_params["inferenceConfig"] = self.model_params

            response = self.client.converse(**converse_params)
            return self._parse_response(response)

        except Exception as e:
            raise LLMGenerationError(
                f"Failed to generate text with Bedrock: {e}"
            ) from e

    @rate_limit_handler_decorator
    def __invoke_v2(
        self,
        input: List[LLMMessage],
        **kwargs: Any,
    ) -> LLMResponse:
        """Sends messages to the LLM and returns a response (V2 interface).

        Args:
            input: List of LLMMessage objects.
            **kwargs: Additional parameters for the API call.

        Returns:
            LLMResponse: The response from the LLM.

        Raises:
            LLMGenerationError: If text generation fails.
        """
        try:
            messages, system = self.get_messages_v2(input)

            converse_params: Dict[str, Any] = {
                "modelId": self._get_model_identifier(),
                "messages": messages,
            }
            if system:
                converse_params["system"] = system

            merged_params = {**self.model_params, **kwargs}
            if merged_params:
                converse_params["inferenceConfig"] = merged_params

            response = self.client.converse(**converse_params)
            return self._parse_response(response)

        except Exception as e:
            raise LLMGenerationError(
                f"Failed to generate text with Bedrock: {e}"
            ) from e

    @async_rate_limit_handler_decorator
    async def __ainvoke_v1(
        self,
        input: str,
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
    ) -> LLMResponse:
        """Asynchronously sends text to the LLM (V1 interface).

        boto3 is synchronous, so this runs the sync method in an executor.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.__invoke_v1(input, message_history, system_instruction),
        )

    @async_rate_limit_handler_decorator
    async def __ainvoke_v2(
        self,
        input: List[LLMMessage],
        **kwargs: Any,
    ) -> LLMResponse:
        """Asynchronously sends messages to the LLM (V2 interface).

        boto3 is synchronous, so this runs the sync method in an executor.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, lambda: self.__invoke_v2(input, **kwargs)
        )

    def _parse_response(self, response: Dict[str, Any]) -> LLMResponse:
        """Parse the Converse API response into an LLMResponse."""
        output = response.get("output", {})
        message = output.get("message", {})
        content_blocks = message.get("content", [])

        if not content_blocks:
            raise LLMGenerationError("LLM returned empty response.")

        # Extract text from content blocks
        text_parts = []
        for block in content_blocks:
            if "text" in block:
                text_parts.append(block["text"])

        if not text_parts:
            raise LLMGenerationError("LLM returned no text content.")

        return LLMResponse(content="".join(text_parts))

    def get_messages(
        self,
        input: str,
        message_history: Optional[List[LLMMessage]] = None,
    ) -> List[MessageTypeDef]:
        """Construct messages for Converse API (V1 interface)."""
        messages: List[MessageTypeDef] = []

        if message_history:
            try:
                MessageList(messages=cast(list[BaseMessage], message_history))
            except ValidationError as e:
                raise LLMGenerationError(e.errors()) from e

            for msg in message_history:
                if msg["role"] in ("user", "assistant"):
                    messages.append({
                        "role": msg["role"],
                        "content": [{"text": msg["content"]}],
                    })

        # Add the current user input
        messages.append({
            "role": "user",
            "content": [{"text": input}],
        })

        return messages

    def get_system(
        self,
        system_instruction: Optional[str] = None,
    ) -> Optional[List[SystemContentBlockTypeDef]]:
        """Construct system message for Converse API."""
        if system_instruction:
            return [{"text": system_instruction}]
        return None

    def get_messages_v2(
        self,
        input: List[LLMMessage],
    ) -> tuple[List[MessageTypeDef], Optional[List[SystemContentBlockTypeDef]]]:
        """Construct messages for Converse API (V2 interface)."""
        messages: List[MessageTypeDef] = []
        system: Optional[List[SystemContentBlockTypeDef]] = None

        for msg in input:
            if msg["role"] == "system":
                system = [{"text": msg["content"]}]
            elif msg["role"] in ("user", "assistant"):
                messages.append({
                    "role": msg["role"],
                    "content": [{"text": msg["content"]}],
                })
            else:
                raise ValueError(f"Unknown role: {msg['role']}")

        return messages, system

    # Tool calling methods
    @rate_limit_handler_decorator
    def invoke_with_tools(
        self,
        input: str,
        tools: Sequence[Tool],
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
    ) -> ToolCallResponse:
        """Sends text to the LLM with tool definitions.

        Args:
            input: Text sent to the LLM.
            tools: Sequence of Tools for the LLM to choose from.
            message_history: A collection of previous messages.
            system_instruction: Optional system message override.

        Returns:
            ToolCallResponse: The response containing tool calls.

        Raises:
            LLMGenerationError: If anything goes wrong.
        """
        try:
            if isinstance(message_history, MessageHistory):
                message_history = message_history.messages

            messages = self.get_messages(input, message_history)
            system = self.get_system(system_instruction)
            tool_config = self._convert_tools_to_bedrock_format(tools)

            converse_params: Dict[str, Any] = {
                "modelId": self._get_model_identifier(),
                "messages": messages,
                "toolConfig": tool_config,
            }
            if system:
                converse_params["system"] = system
            if self.model_params:
                converse_params["inferenceConfig"] = self.model_params

            response = self.client.converse(**converse_params)
            return self._parse_tool_response(response)

        except Exception as e:
            raise LLMGenerationError(
                f"Failed to invoke with tools on Bedrock: {e}"
            ) from e

    @async_rate_limit_handler_decorator
    async def ainvoke_with_tools(
        self,
        input: str,
        tools: Sequence[Tool],
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
    ) -> ToolCallResponse:
        """Asynchronously sends text to the LLM with tool definitions.

        boto3 is synchronous, so this runs the sync method in an executor.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.invoke_with_tools(
                input, tools, message_history, system_instruction
            ),
        )

    def _convert_tools_to_bedrock_format(
        self, tools: Sequence[Tool]
    ) -> Dict[str, Any]:
        """Convert Tool objects to Bedrock's toolConfig format."""
        bedrock_tools = []
        for tool in tools:
            tool_spec = {
                "toolSpec": {
                    "name": tool.get_name(),
                    "description": tool.get_description(),
                    "inputSchema": {
                        "json": tool.get_parameters(),
                    },
                }
            }
            bedrock_tools.append(tool_spec)

        return {"tools": bedrock_tools}

    def _parse_tool_response(self, response: Dict[str, Any]) -> ToolCallResponse:
        """Parse the Converse API response for tool calls."""
        output = response.get("output", {})
        message = output.get("message", {})
        content_blocks = message.get("content", [])

        tool_calls = []
        text_content = None

        for block in content_blocks:
            if "toolUse" in block:
                tool_use = block["toolUse"]
                tool_calls.append(
                    ToolCall(
                        name=tool_use["name"],
                        arguments=tool_use.get("input", {}),
                    )
                )
            elif "text" in block:
                text_content = block["text"]

        return ToolCallResponse(tool_calls=tool_calls, content=text_content)
