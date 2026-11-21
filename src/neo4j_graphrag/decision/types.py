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

from typing import Optional

from pydantic import BaseModel


class DecisionUsage(BaseModel):
    """The usage that the provider reports for one decision call.

    Attributes:
        input_tokens (Optional[int]): The number of tokens that the model read, or
            ``None`` if the provider doesn't report it.
        output_tokens (Optional[int]): The number of tokens that the model wrote, or
            ``None`` if the provider doesn't report it.
        cost (Optional[float]): The cost of the call in USD, as billed by the provider,
            or ``None`` if the provider doesn't report it.
    """

    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cost: Optional[float] = None


class DecisionResponse(BaseModel):
    """The response from a decision model.

    Attributes:
        choice (str): The key of the chosen option.
        probabilities (dict[str, float]): The probability of every option, by key.
        confidence (Optional[float]): The model's confidence in its choice, if it
            reports one.
        model (Optional[str]): The exact model version that answered the call, if
            the provider reports it.
        usage (Optional[DecisionUsage]): The tokens and cost of the call, if the
            provider reports them.
    """

    choice: str
    probabilities: dict[str, float]
    confidence: Optional[float] = None
    model: Optional[str] = None
    usage: Optional[DecisionUsage] = None
