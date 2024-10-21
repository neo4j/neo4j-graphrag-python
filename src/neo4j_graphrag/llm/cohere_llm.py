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

from typing import Any, Optional

from neo4j_graphrag.exceptions import LLMGenerationError
from neo4j_graphrag.llm.base import LLMInterface
from neo4j_graphrag.llm.types import LLMResponse

try:
    import cohere
    from cohere.core import ApiError
except ImportError:
    cohere = None  # type: ignore
    ApiError = Exception  # type: ignore[assignment, misc]


class CohereLLM(LLMInterface):
    """Interface for large language models on the Cohere platform

    Args:
        model_name (str, optional): Name of the LLM to use. Defaults to "gemini-1.5-flash-001".
        model_params (Optional[dict], optional): Additional parameters passed to the model when text is sent to it. Defaults to None.
        **kwargs (Any): Arguments passed to the model when for the class is initialised. Defaults to None.

    Raises:
        LLMGenerationError: If there's an error generating the response from the model.

    Example:

    .. code-block:: python

        from neo4j_graphrag.llm import CohereLLM

        llm = CohereLLM(api_key="...")
        llm.invoke("Say something")
    """

    def __init__(
        self,
        model_name: str = "",
        model_params: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        if cohere is None:
            raise ImportError(
                "Could not import cohere python client. "
                "Please install it with `pip install cohere`."
            )
        super().__init__(model_name, model_params)
        self.client = cohere.Client(**kwargs)
        self.async_client = cohere.AsyncClient(**kwargs)

    def invoke(self, input: str) -> LLMResponse:
        """Sends text to the LLM and returns a response.

        Args:
            input (str): The text to send to the LLM.

        Returns:
            LLMResponse: The response from the LLM.
        """
        try:
            res = self.client.chat(
                message=input,
                model=self.model_name,
            )
        except ApiError as e:
            raise LLMGenerationError(e)
        return LLMResponse(
            content=res.text,
        )

    async def ainvoke(self, input: str) -> LLMResponse:
        """Asynchronously sends text to the LLM and returns a response.

        Args:
            input (str): The text to send to the LLM.

        Returns:
            LLMResponse: The response from the LLM.
        """
        try:
            res = await self.async_client.chat(
                message=input,
                model=self.model_name,
            )
        except ApiError as e:
            raise LLMGenerationError(e)
        return LLMResponse(
            content=res.text,
        )
