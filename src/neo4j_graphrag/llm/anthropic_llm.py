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

from typing import (
    TYPE_CHECKING,
    Any,
    Iterable,
    List,
    Optional,
    Type,
    Union,
    cast,
)

from pydantic import BaseModel, ValidationError

from neo4j_graphrag.exceptions import LLMGenerationError
from neo4j_graphrag.llm.base import LLMBase
from neo4j_graphrag.llm.types import (
    BaseMessage,
    LLMResponse,
    LLMUsage,
    MessageList,
    UserMessage,
)
from neo4j_graphrag.message_history import MessageHistory
from neo4j_graphrag.types import LLMMessage
from neo4j_graphrag.utils.rate_limit import (
    RateLimitHandler,
)
from neo4j_graphrag.utils.rate_limit import (
    async_rate_limit_handler as async_rate_limit_handler_decorator,
)
from neo4j_graphrag.utils.rate_limit import (
    rate_limit_handler as rate_limit_handler_decorator,
)

if TYPE_CHECKING:
    from anthropic import NotGiven
    from anthropic.types.message_param import MessageParam


# pylint: disable=redefined-builtin, arguments-differ, raise-missing-from, no-else-return, import-outside-toplevel
class AnthropicLLM(LLMBase):
    supports_structured_output: bool = True
    """Interface for large language models on Anthropic

    Args:
        model_name (str): Name of the LLM to use.
        model_params (Optional[dict], optional): Additional parameters for LLMInterface(V1) passed to the model when text is sent to it. Defaults to None.
        system_instruction: Optional[str], optional): Additional instructions for setting the behavior and context for the model in a conversation. Defaults to None.
        rate_limit_handler (Optional[RateLimitHandler], optional): Handler for managing rate limits for LLMInterface(V1). Defaults to None.
        **kwargs (Any): Arguments passed to the model when for the class is initialised. Defaults to None.

    Raises:
        LLMGenerationError: If there's an error generating the response from the model.

    Example:

    .. code-block:: python

        from neo4j_graphrag.llm import AnthropicLLM

        llm = AnthropicLLM(
            model_name="claude-3-opus-20240229",
            model_params={"max_tokens": 1000},
            api_key="sk...",   # can also be read from env vars
        )
        llm.invoke("Who is the mother of Paul Atreides?")
    """

    def __init__(
        self,
        model_name: str,
        model_params: Optional[dict[str, Any]] = None,
        rate_limit_handler: Optional[RateLimitHandler] = None,
        **kwargs: Any,
    ):
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                """Could not import Anthropic Python client.
                Please install it with `pip install "neo4j-graphrag[anthropic]"`."""
            )
        LLMBase.__init__(
            self,
            model_name=model_name,
            model_params=model_params or {},
            rate_limit_handler=rate_limit_handler,
            **kwargs,
        )
        self.anthropic = anthropic
        self.client = anthropic.Anthropic(**kwargs)
        self.async_client = anthropic.AsyncAnthropic(**kwargs)

    def invoke(
        self,
        input: Union[str, List[LLMMessage]],
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
        response_format: Optional[Union[Type[BaseModel], dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        if isinstance(input, str):
            return self.__invoke_v1(input, message_history, system_instruction)
        elif isinstance(input, list):
            return self.__invoke_v2(input, response_format=response_format, **kwargs)
        else:
            raise ValueError(f"Invalid input type for invoke method - {type(input)}")

    async def ainvoke(
        self,
        input: Union[str, List[LLMMessage]],
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
        response_format: Optional[Union[Type[BaseModel], dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        if isinstance(input, str):
            return await self.__ainvoke_v1(input, message_history, system_instruction)
        elif isinstance(input, list):
            return await self.__ainvoke_v2(
                input, response_format=response_format, **kwargs
            )
        else:
            raise ValueError(f"Invalid input type for ainvoke method - {type(input)}")

    # implementaions
    @rate_limit_handler_decorator
    def __invoke_v1(
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
        try:
            if isinstance(message_history, MessageHistory):
                message_history = message_history.messages
            messages = self.get_messages(input, message_history)
            response = self.client.messages.create(
                model=self.model_name,
                system=system_instruction or self.anthropic.NOT_GIVEN,
                messages=messages,
                **self.model_params,
            )
            response_content = response.content
            if response_content and len(response_content) > 0:
                text = response_content[0].text
            else:
                raise LLMGenerationError("LLM returned empty response.")
            usage = LLMUsage(
                request_tokens=response.usage.input_tokens,
                response_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens,
            )
            return LLMResponse(content=text, usage=usage)
        except self.anthropic.APIError as e:
            raise LLMGenerationError(e)

    @rate_limit_handler_decorator
    def __invoke_v2(
        self,
        input: List[LLMMessage],
        response_format: Optional[Union[Type[BaseModel], dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        try:
            system_instruction, messages = self.get_messages_v2(input)
            if response_format is not None:
                tool_name, tools, tool_choice = self._build_tool_choice(response_format)
                response = self.client.messages.create(
                    model=self.model_name,
                    system=system_instruction,
                    messages=messages,
                    tools=tools,
                    tool_choice=tool_choice,
                    **self.model_params,
                    **kwargs,
                )
                text = self._extract_tool_result(response, tool_name)
            else:
                response = self.client.messages.create(
                    model=self.model_name,
                    system=system_instruction,
                    messages=messages,
                    **self.model_params,
                    **kwargs,
                )
                response_content = response.content
                if response_content and len(response_content) > 0:
                    text = response_content[0].text
                else:
                    raise LLMGenerationError("LLM returned empty response.")
            usage = LLMUsage(
                request_tokens=response.usage.input_tokens,
                response_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens,
            )
            return LLMResponse(content=text, usage=usage)
        except self.anthropic.APIError as e:
            raise LLMGenerationError(e)

    @async_rate_limit_handler_decorator
    async def __ainvoke_v1(
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
            messages = self.get_messages(input, message_history)
            response = await self.async_client.messages.create(
                model=self.model_name,
                system=system_instruction or self.anthropic.NOT_GIVEN,
                messages=messages,
                **self.model_params,
            )
            response_content = response.content
            if response_content and len(response_content) > 0:
                text = response_content[0].text
            else:
                raise LLMGenerationError("LLM returned empty response.")
            usage = LLMUsage(
                request_tokens=response.usage.input_tokens,
                response_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens,
            )
            return LLMResponse(content=text, usage=usage)
        except self.anthropic.APIError as e:
            raise LLMGenerationError(e)

    @async_rate_limit_handler_decorator
    async def __ainvoke_v2(
        self,
        input: List[LLMMessage],
        response_format: Optional[Union[Type[BaseModel], dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Asynchronously sends text to the LLM and returns a response.

        Args:
            input (List[LLMMessage]): The messages to send to the LLM.
            response_format (Optional[Union[Type[BaseModel], dict[str, Any]]]): Optional
                response format. Can be a Pydantic model class for structured output
                or a dict containing a JSON schema.

        Returns:
            LLMResponse: The response from the LLM.
        """
        try:
            system_instruction, messages = self.get_messages_v2(input)
            if response_format is not None:
                tool_name, tools, tool_choice = self._build_tool_choice(response_format)
                response = await self.async_client.messages.create(
                    model=self.model_name,
                    system=system_instruction,
                    messages=messages,
                    tools=tools,
                    tool_choice=tool_choice,
                    **self.model_params,
                    **kwargs,
                )
                text = self._extract_tool_result(response, tool_name)
            else:
                response = await self.async_client.messages.create(
                    model=self.model_name,
                    system=system_instruction,
                    messages=messages,
                    **self.model_params,
                    **kwargs,
                )
                response_content = response.content
                if response_content and len(response_content) > 0:
                    text = response_content[0].text
                else:
                    raise LLMGenerationError("LLM returned empty response.")
            usage = LLMUsage(
                request_tokens=response.usage.input_tokens,
                response_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens,
            )
            return LLMResponse(content=text, usage=usage)
        except self.anthropic.APIError as e:
            raise LLMGenerationError(e)

    async def aclose(self) -> None:
        self.client.close()
        await self.async_client.close()

    # subsidiary methods
    def _build_tool_choice(
        self,
        response_format: Union[Type[BaseModel], dict[str, Any]],
    ) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
        """Build Anthropic tool-forcing args for structured output.

        Anthropic implements structured output by defining a single tool whose
        input_schema is the desired JSON schema, then forcing the model to call
        it via tool_choice={"type": "tool", "name": <name>}.

        Args:
            response_format: A Pydantic BaseModel subclass or a raw JSON schema dict.

        Returns:
            A tuple of (tool_name, tools_list, tool_choice_dict).
        """
        if isinstance(response_format, type) and issubclass(response_format, BaseModel):
            tool_name = response_format.__name__
            schema = response_format.model_json_schema()
        else:
            tool_name = response_format.get("title", "structured_output")
            schema = response_format
        tools = [
            {
                "name": tool_name,
                "description": f"Return a {tool_name} object.",
                "input_schema": schema,
            }
        ]
        tool_choice: dict[str, Any] = {"type": "tool", "name": tool_name}
        return tool_name, tools, tool_choice

    def _extract_tool_result(self, response: Any, tool_name: str) -> str:
        """Extract JSON string from a tool-use response block.

        Args:
            response: The raw Anthropic API response.
            tool_name: The name of the forced tool.

        Returns:
            JSON string of the tool input.

        Raises:
            LLMGenerationError: If no tool_use block is found.
        """
        import json
        for block in response.content:
            if getattr(block, "type", None) == "tool_use" and block.name == tool_name:
                return json.dumps(block.input)
        raise LLMGenerationError(
            f"AnthropicLLM structured output: no tool_use block for '{tool_name}' in response."
        )

    def get_messages(
        self,
        input: str,
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
    ) -> Iterable[MessageParam]:
        """Constructs the message list for the LLM from the input and message history."""
        messages: list[dict[str, str]] = []
        if message_history:
            if isinstance(message_history, MessageHistory):
                message_history = message_history.messages
            try:
                MessageList(messages=cast(list[BaseMessage], message_history))
            except ValidationError as e:
                raise LLMGenerationError(e.errors()) from e
            messages.extend(cast(Iterable[dict[str, Any]], message_history))
        messages.append(UserMessage(content=input).model_dump())
        return messages  # type: ignore

    def get_messages_v2(
        self,
        input: list[LLMMessage],
    ) -> tuple[Union[str, NotGiven], Iterable[MessageParam]]:
        """Constructs the message list for the LLM from the input."""
        messages: list[MessageParam] = []
        system_instruction: Union[str, NotGiven] = self.anthropic.NOT_GIVEN
        for i in input:
            if i["role"] == "system":
                system_instruction = i["content"]
            else:
                if i["role"] not in ("user", "assistant"):
                    raise ValueError(f"Unknown role: {i['role']}")
                messages.append(
                    self.anthropic.types.MessageParam(
                        role=i["role"],
                        content=i["content"],
                    )
                )
        return system_instruction, messages
