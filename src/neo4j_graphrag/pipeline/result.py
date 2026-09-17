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
"""Result type for partial-failure-safe pipeline stages.

Defines the two variants produced by ``_safe`` operators on
:class:`~neo4j_graphrag.pipeline.pipeline.Pipeline`:

* :class:`Ok` – a successful value wrapped in the result stream.
* :class:`Err` – a captured exception that stays in the stream until
  explicitly handled via
  :meth:`~neo4j_graphrag.pipeline.pipeline.ResultPipeline.on_error` or
  :meth:`~neo4j_graphrag.pipeline.pipeline.ResultPipeline.filter_ok`.

Having a shared :class:`Result` base class lets callers write
``isinstance(item, Result)`` guards and makes the contract visible in
type annotations.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Generic, TypeVar

__all__ = ["Result", "Ok", "Err"]

T = TypeVar("T")


class Result(ABC):
    """Abstract base for :class:`Ok` and :class:`Err`.

    Not intended for direct instantiation.  Use the concrete subclasses
    produced by the ``_safe`` pipeline operators.
    """


@dataclass(frozen=True)
class Ok(Result, Generic[T]):
    """Wraps a successful value in a Result stream."""

    value: T


@dataclass(frozen=True)
class Err(Result):
    """Wraps a failed computation in a Result stream.

    Attributes:
        exception: The exception that caused the failure.
    """

    exception: Exception
