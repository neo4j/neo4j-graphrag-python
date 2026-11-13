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

from abc import ABC, abstractmethod
from typing import Any, Optional, Union

from neo4j_graphrag.utils.rate_limit import (
    DEFAULT_RATE_LIMIT_HANDLER,
    RateLimitHandler,
)

from .types import DecisionResponse

DecisionState = Union[str, dict[str, Any], list[Any]]
"""The situation that a decision is made in. It can be text, or any object or list that
can be serialized to JSON."""


class DecisionModelInterface(ABC):
    """Interface for decision models.

    A decision model does not generate text. It reads a state and a set of options,
    and it returns a probability for every option, together with the option that it
    chose. Any decision model that you pass to a retriever must implement this
    interface.

    Args:
        model_name (str): The name of the decision model.
        max_options (int): The largest number of options the model accepts in one call.
            Defaults to 255.
        rate_limit_handler (Optional[RateLimitHandler]): Handler for rate limiting.
            Defaults to retry with exponential backoff.
    """

    def __init__(
        self,
        model_name: str,
        max_options: int = 255,
        rate_limit_handler: Optional[RateLimitHandler] = None,
    ):
        if max_options < 2:
            raise ValueError("max_options must be at least 2")
        self.model_name = model_name
        self.max_options = max_options
        if rate_limit_handler is not None:
            self._rate_limit_handler = rate_limit_handler
        else:
            self._rate_limit_handler = DEFAULT_RATE_LIMIT_HANDLER

    @abstractmethod
    def decide(
        self,
        state: DecisionState,
        options: dict[str, str],
        instructions: str,
    ) -> DecisionResponse:
        """Choose one of the options.

        Args:
            state (DecisionState): The situation that the decision is made in.
            options (dict[str, str]): The options, as a map from each key to its
                description. There can be at most ``max_options`` options.
            instructions (str): What the choice is for.

        Returns:
            DecisionResponse: The chosen key and the probability of every option.

        Raises:
            DecisionModelError: If the response is invalid, or if the chosen key is not
                one of the options. A temporary failure raises
                :class:`~neo4j_graphrag.exceptions.RetryableError` instead. When a rate
                limit handler stops retrying, it raises ``tenacity.RetryError``.
        """

    @abstractmethod
    async def adecide(
        self,
        state: DecisionState,
        options: dict[str, str],
        instructions: str,
    ) -> DecisionResponse:
        """Asynchronously choose one of the options.

        Args:
            state (DecisionState): The situation that the decision is made in.
            options (dict[str, str]): The options, as a map from each key to its
                description. There can be at most ``max_options`` options.
            instructions (str): What the choice is for.

        Returns:
            DecisionResponse: The chosen key and the probability of every option.

        Raises:
            DecisionModelError: If the response is invalid, or if the chosen key is not
                one of the options. A temporary failure raises
                :class:`~neo4j_graphrag.exceptions.RetryableError` instead. When a rate
                limit handler stops retrying, it raises ``tenacity.RetryError``.
        """
