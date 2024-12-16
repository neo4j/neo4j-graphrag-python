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
from typing import Any, Optional, Sequence, TYPE_CHECKING

from pydantic import ValidationError

from neo4j_graphrag.exceptions import LLMGenerationError

from .base import LLMInterface
from .types import LLMResponse, SystemMessage, UserMessage, MessageList

if TYPE_CHECKING:
    import ollama
    from ollama import Message


class OllamaLLM(LLMInterface):
    def __init__(
        self,
        model_name: str,
        model_params: Optional[dict[str, Any]] = None,
        system_instruction: Optional[str] = None,
        **kwargs: Any,
    ):
        try:
            import ollama
        except ImportError:
            raise ImportError(
                "Could not import ollama Python client. "
                "Please install it with `pip install ollama`."
            )
        super().__init__(model_name, model_params, system_instruction, **kwargs)
        self.ollama = ollama
        self.client = ollama.Client(
            **kwargs,
        )
        self.async_client = ollama.AsyncClient(
            **kwargs,
        )

    def get_messages(
        self, input: str, chat_history: Optional[list[Any]] = None
    ) -> Sequence[Message]:
        messages = []
        if self.system_instruction:
            messages.append(SystemMessage(content=self.system_instruction).model_dump())
        if chat_history:
            try:
                MessageList(messages=chat_history)
            except ValidationError as e:
                raise LLMGenerationError(e.errors()) from e
            messages.extend(chat_history)
        messages.append(UserMessage(content=input).model_dump())
        return messages

    def invoke(
        self, input: str, chat_history: Optional[list[Any]] = None
    ) -> LLMResponse:
        try:
            response = self.client.chat(
                model=self.model_name,
                messages=self.get_messages(input, chat_history),
                options=self.model_params,
            )
            content = response.message.content or ""
            return LLMResponse(content=content)
        except self.ollama.ResponseError as e:
            raise LLMGenerationError(e)

    async def ainvoke(
        self, input: str, chat_history: Optional[list[Any]] = None
    ) -> LLMResponse:
        try:
            response = await self.async_client.chat(
                model=self.model_name,
                messages=self.get_messages(input, chat_history),
                options=self.model_params,
            )
            content = response.message.content or ""
            return LLMResponse(content=content)
        except self.ollama.ResponseError as e:
            raise LLMGenerationError(e)
