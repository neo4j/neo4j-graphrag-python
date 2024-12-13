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
from typing import Any, Optional

from neo4j_graphrag.exceptions import LLMGenerationError

from .base import LLMInterface
from .types import LLMResponse


class OllamaLLM(LLMInterface):
    def __init__(
        self,
        model_name: str,
        model_params: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ):
        try:
            import ollama
        except ImportError:
            raise ImportError(
                "Could not import ollama Python client. "
                "Please install it with `pip install ollama`."
            )
        super().__init__(model_name, model_params, **kwargs)
        self.ollama = ollama
        self.client = ollama.Client(
            **kwargs,
        )
        self.async_client = ollama.AsyncClient(
            **kwargs,
        )

    def invoke(self, input: str) -> LLMResponse:
        try:
            response = self.client.chat(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": input,
                    },
                ],
            )
            content = response.message.content or ""
            return LLMResponse(content=content)
        except self.ollama.ResponseError as e:
            raise LLMGenerationError(e)

    async def ainvoke(self, input: str) -> LLMResponse:
        try:
            response = await self.async_client.chat(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": input,
                    },
                ],
            )
            content = response.message.content or ""
            return LLMResponse(content=content)
        except self.ollama.ResponseError as e:
            raise LLMGenerationError(e)
