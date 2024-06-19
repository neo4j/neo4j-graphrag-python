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
from abc import ABC, abstractmethod
from typing import Optional, Any

from .types import LLMResponse


class LLMInterface(ABC):
    """Interface for large language models."""

    def __init__(
        self,
        model_name: str,
        model_params: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ):
        self.model_name = model_name
        self.model_params = model_params or {}

    @abstractmethod
    def invoke(self, input: str) -> LLMResponse:
        """Sends a text input to the LLM and retrieves a response.

        Args:
            input (str): Text sent to the LLM

        Returns:
            LLMResponse: The response from the LLM.

        Raises:
            LLMGenerationError: If anything goes wrong.
        """
