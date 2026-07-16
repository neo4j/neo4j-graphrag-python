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

import json
import warnings
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

# 3rd party dependencies
try:
    import httpx
except ImportError:
    httpx = None  # type: ignore[assignment]

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
    from anthropic import Omit
    from anthropic.types.message_param import MessageParam


# ---------------------------------------------------------------------------
# TEMPORARY / INTERMEDIATE FIX -- REMOVE ONCE CROSS-PROVIDER STRICT JSON SCHEMA
# HANDLING IS ADDED.
#
# Anthropic structured output uses constrained decoding and only accepts a
# closed JSON Schema subset: every object must set ``additionalProperties: false``
# and open-ended maps (Pydantic ``dict[str, X]`` -> ``additionalProperties`` as a
# *schema*) are rejected with a 400 ("additionalProperties: object is not
# supported"). Naively forcing ``additionalProperties: false`` would instead make
# those maps un-fillable and silently drop every property value.
#
# To fix the 400 *without* dropping properties, and without touching the shared
# components/other providers, we transform open maps into closed key/value-pair
# arrays on the way out (:func:`_to_anthropic_schema`) and convert them back to
# maps on the way in (:func:`_restore_open_maps`), so the returned content stays
# byte-compatible with the caller's Pydantic model (e.g. ``Neo4jGraph``).
#
# When a proper, cross-provider strict-JSON-schema mechanism lands, delete this
# whole block, the two ``_restore_open_maps`` call sites in ``__invoke_v2`` /
# ``__ainvoke_v2``, and restore ``_build_output_config`` to passing the raw
# ``model_json_schema()`` through.
# ---------------------------------------------------------------------------


def _is_open_map(schema: dict[str, Any]) -> bool:
    """True if *schema* is an open-ended map (``dict[str, X]``) rather than a
    fixed-property object."""
    return (
        schema.get("type") == "object"
        and isinstance(schema.get("additionalProperties"), dict)
        and not schema.get("properties")
    )


def _to_anthropic_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Rewrite a JSON schema into Anthropic's constrained-decoding subset.

    Open maps become closed ``[{"key": ..., "value": ...}]`` arrays, and every
    fixed-property object gets ``additionalProperties: false`` plus a full
    ``required`` list.
    """
    schema = dict(schema)
    if _is_open_map(schema):
        value_schema = _to_anthropic_schema(schema["additionalProperties"])
        return {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"key": {"type": "string"}, "value": value_schema},
                "required": ["key", "value"],
                "additionalProperties": False,
            },
        }
    if schema.get("type") == "object" and "properties" in schema:
        schema["properties"] = {
            key: _to_anthropic_schema(prop)
            for key, prop in schema["properties"].items()
        }
        schema["additionalProperties"] = False
        schema["required"] = list(schema["properties"].keys())
    if "items" in schema:
        schema["items"] = _to_anthropic_schema(schema["items"])
    for combinator in ("anyOf", "oneOf", "allOf"):
        if combinator in schema:
            schema[combinator] = [
                _to_anthropic_schema(variant) for variant in schema[combinator]
            ]
    if "$defs" in schema:
        schema["$defs"] = {
            name: _to_anthropic_schema(def_schema)
            for name, def_schema in schema["$defs"].items()
        }
    return schema


def _resolve_ref(schema: dict[str, Any], defs: dict[str, Any]) -> dict[str, Any]:
    """Resolve a local ``$ref`` against *defs*, if present."""
    ref = schema.get("$ref")
    if isinstance(ref, str):
        return cast("dict[str, Any]", defs.get(ref.split("/")[-1], {}))
    return schema


def _restore_open_maps(value: Any, schema: dict[str, Any], defs: dict[str, Any]) -> Any:
    """Convert key/value-pair arrays produced for Anthropic back into maps.

    Walks *value* alongside the caller's *original* (untransformed) JSON schema,
    so empty maps (``[]`` -> ``{}``) and genuine empty arrays are disambiguated
    correctly.
    """
    schema = _resolve_ref(schema, defs)
    if _is_open_map(schema) and isinstance(value, list):
        value_schema = schema["additionalProperties"]
        return {
            item["key"]: _restore_open_maps(item["value"], value_schema, defs)
            for item in value
        }
    if schema.get("type") == "object" and isinstance(value, dict):
        properties = schema.get("properties", {})
        return {
            key: (
                _restore_open_maps(val, properties[key], defs)
                if key in properties
                else val
            )
            for key, val in value.items()
        }
    if schema.get("type") == "array" and isinstance(value, list):
        item_schema = schema.get("items", {})
        return [_restore_open_maps(item, item_schema, defs) for item in value]
    return value


# pylint: disable=redefined-builtin, arguments-differ, raise-missing-from, no-else-return, import-outside-toplevel
class AnthropicLLM(LLMBase):
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

    supports_structured_output: bool = True

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
        http_client = kwargs.pop("http_client", None)
        sync_params = kwargs.copy()
        async_params = kwargs.copy()
        if httpx is not None and isinstance(http_client, httpx.Client):
            sync_params["http_client"] = http_client
        elif httpx is not None and isinstance(http_client, httpx.AsyncClient):
            async_params["http_client"] = http_client
        elif http_client is not None:
            warnings.warn(
                f"Invalid http_client type (got {type(http_client)}, expected httpx.Client or httpx.AsyncClient). Using default client.",
                stacklevel=2,
            )
        self.client = anthropic.Anthropic(**sync_params)
        self.async_client = anthropic.AsyncAnthropic(**async_params)

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
                system=system_instruction or self.anthropic.omit,
                messages=messages,
                **self.model_params,
            )
            text = self._extract_text(response)
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
                kwargs["output_config"] = self._build_output_config(response_format)
            response = self.client.messages.create(
                model=self.model_name,
                system=system_instruction,
                messages=messages,
                **self.model_params,
                **kwargs,
            )
            text = self._extract_text(response)
            # INTERMEDIATE FIX (see module-level note): remove with the rest of
            # the open-map workaround once cross-provider strict schema handling
            # is added.
            text = self._restore_structured_output(text, response_format)
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
                system=system_instruction or self.anthropic.omit,
                messages=messages,
                **self.model_params,
            )
            text = self._extract_text(response)
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
                kwargs["output_config"] = self._build_output_config(response_format)
            response = await self.async_client.messages.create(
                model=self.model_name,
                system=system_instruction,
                messages=messages,
                **self.model_params,
                **kwargs,
            )
            text = self._extract_text(response)
            # INTERMEDIATE FIX (see module-level note): remove with the rest of
            # the open-map workaround once cross-provider strict schema handling
            # is added.
            text = self._restore_structured_output(text, response_format)
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
    @staticmethod
    def _extract_text(response: Any) -> str:
        """Extracts the text of the first content block from an Anthropic response.

        The Anthropic SDK returns a union of content block types (text, thinking,
        tool use, etc.), only some of which expose a ``text`` attribute. This
        guards against empty responses and non-text blocks.

        Args:
            response: The response object returned by ``messages.create``.

        Returns:
            The text of the first content block.

        Raises:
            LLMGenerationError: If the response is empty or the first block is
                not a text block.
        """
        content = response.content
        if not content:
            raise LLMGenerationError("LLM returned empty response.")
        block = content[0]
        text = getattr(block, "text", None)
        if not isinstance(text, str):
            raise LLMGenerationError(
                f"Expected a text block in the response, got {type(block).__name__}."
            )
        return text

    @staticmethod
    def _build_output_config(
        response_format: Union[Type[BaseModel], dict[str, Any]],
    ) -> dict[str, Any]:
        """Builds the Anthropic output_config for structured output.

        Anthropic exposes a first-class structured-output API via output_config
        with type "json_schema", which uses constrained decoding to guarantee
        schema-conforming output.

        Args:
            response_format: A Pydantic BaseModel subclass, or a dict already
                matching Anthropic's output_config schema.

        Returns:
            A dict suitable for the `output_config` kwarg to `messages.create`.
        """
        if isinstance(response_format, type) and issubclass(response_format, BaseModel):
            # INTERMEDIATE FIX (see module-level note): transform open maps into
            # Anthropic-compatible closed key/value schemas. Remove when
            # cross-provider strict JSON schema handling is added and pass
            # ``response_format.model_json_schema()`` through directly.
            schema = _to_anthropic_schema(response_format.model_json_schema())
            return {"format": {"type": "json_schema", "schema": schema}}
        return response_format

    @staticmethod
    def _restore_structured_output(
        text: str,
        response_format: Optional[Union[Type[BaseModel], dict[str, Any]]],
    ) -> str:
        """Reverse :func:`_to_anthropic_schema` on the response text.

        INTERMEDIATE FIX (see module-level note): converts the key/value-pair
        arrays Anthropic was constrained to emit back into the open maps expected
        by the caller's Pydantic model, so ``content`` round-trips unchanged.
        Remove when cross-provider strict JSON schema handling is added.
        """
        if not (
            isinstance(response_format, type) and issubclass(response_format, BaseModel)
        ):
            return text
        original_schema = response_format.model_json_schema()
        defs = original_schema.get("$defs", {})
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return text
        restored = _restore_open_maps(data, original_schema, defs)
        return json.dumps(restored)

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
        return cast("Iterable[MessageParam]", messages)

    def get_messages_v2(
        self,
        input: list[LLMMessage],
    ) -> tuple[Union[str, Omit], Iterable[MessageParam]]:
        """Constructs the message list for the LLM from the input."""
        messages: list[MessageParam] = []
        system_instruction: Union[str, Omit] = self.anthropic.omit
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
