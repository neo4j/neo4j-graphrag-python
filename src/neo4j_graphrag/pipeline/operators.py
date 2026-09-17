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
"""Operator nodes for the embedded pipeline DSL.

Operators are **pure data**: each node records the previous node (``prev``)
and the parameters of the stage (the function to apply, batch size, …).
They contain no evaluation logic — evaluating a pipeline is the job of an
:class:`~neo4j_graphrag.pipeline.interpreter.Interpreter`.

A pipeline definition is a singly-linked list of operators starting at a
:class:`SourceOp` and ending at the pipeline's *tail* node.  Builders
(:class:`~neo4j_graphrag.pipeline.pipeline.Pipeline` and
:class:`~neo4j_graphrag.pipeline.pipeline.ResultPipeline`) append nodes;
they never execute anything.

Node naming conventions:

- Plain names (``Map``, ``FlatMap``, …) apply a function directly;
  exceptions propagate and abort the stream.
- ``Try*`` nodes capture per-item exceptions as
  :class:`~neo4j_graphrag.pipeline.result.Err` values, turning the stream
  into a stream of :class:`~neo4j_graphrag.pipeline.result.Ok` /
  :class:`~neo4j_graphrag.pipeline.result.Err`.
- ``*Ok`` nodes operate on the inner value of ``Ok`` items in a result
  stream, passing ``Err`` items through unchanged.
- ``*AsyncChunked`` nodes apply an async function concurrently in chunks
  of ``map_batch_size`` items.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from typing import Any

from neo4j_graphrag.pipeline.result import Err
from neo4j_graphrag.pipeline.sink import Sink
from neo4j_graphrag.pipeline.source import Source

__all__ = [
    "Operator",
    "SourceOp",
    "Map",
    "FlatMap",
    "Filter",
    "Take",
    "TakeWhile",
    "Skip",
    "Grouped",
    "ReduceOp",
    "MapAsyncChunked",
    "TryMap",
    "TryMapAsyncChunked",
    "MapOk",
    "FlattenOk",
    "TryMapOk",
    "TryMapOkAsyncChunked",
    "FilterOk",
    "OnError",
    "SinkOp",
]


@dataclass
class Operator:
    """Base class for all pipeline operators.

    Attributes:
        prev: The previous operator in the chain, or ``None`` for the
            :class:`SourceOp` root.
    """

    prev: Operator | None


@dataclass
class SourceOp(Operator):
    """Root of every pipeline: emits elements from a ``Source``."""

    source: Source[Any]


# ---------------------------------------------------------------------------
# Synchronous operators
# ---------------------------------------------------------------------------


@dataclass
class Map(Operator):
    """Apply ``func`` to every element (1-to-1)."""

    func: Callable[[Any], Any]


@dataclass
class FlatMap(Operator):
    """Apply ``func`` and flatten one level (1-to-many)."""

    func: Callable[[Any], Iterable[Any]]


@dataclass
class Filter(Operator):
    """Keep only elements for which ``predicate`` is ``True``."""

    predicate: Callable[[Any], bool]


@dataclass
class Take(Operator):
    """Take the first ``n`` elements."""

    n: int


@dataclass
class TakeWhile(Operator):
    """Take elements while ``predicate`` is ``True``, then stop."""

    predicate: Callable[[Any], bool]


@dataclass
class Skip(Operator):
    """Skip the first ``n`` elements."""

    n: int


@dataclass
class Grouped(Operator):
    """Collect elements into batches of up to ``size`` items."""

    size: int


@dataclass
class ReduceOp(Operator):
    """Fold all elements into a single value (emitted as one element)."""

    zero: Any
    combine: Callable[[Any, Any], Any]


# ---------------------------------------------------------------------------
# Async operators
# ---------------------------------------------------------------------------


@dataclass
class MapAsyncChunked(Operator):
    """Apply async ``func`` concurrently in chunks of ``map_batch_size``."""

    func: Callable[[Any], Awaitable[Any]]
    map_batch_size: int


# ---------------------------------------------------------------------------
# Partial-failure safe operators (plain stream -> result stream)
# ---------------------------------------------------------------------------


@dataclass
class TryMap(Operator):
    """Like :class:`Map`, but per-item exceptions become ``Err`` values."""

    func: Callable[[Any], Any]


@dataclass
class TryMapAsyncChunked(Operator):
    """Like :class:`MapAsyncChunked`, capturing per-item exceptions as ``Err``."""

    func: Callable[[Any], Awaitable[Any]]
    map_batch_size: int


# ---------------------------------------------------------------------------
# Result-stream combinators (result stream -> result stream)
# ---------------------------------------------------------------------------


@dataclass
class MapOk(Operator):
    """Apply ``func`` to the value inside each ``Ok``; ``Err`` passes through."""

    func: Callable[[Any], Any]


@dataclass
class FlattenOk(Operator):
    """Expand each ``Ok`` holding an iterable into one ``Ok`` per item."""


@dataclass
class TryMapOk(Operator):
    """Like :class:`MapOk`, but new exceptions are captured as ``Err``."""

    func: Callable[[Any], Any]


@dataclass
class TryMapOkAsyncChunked(Operator):
    """Async chunked mapping of ``Ok`` values; ``Err`` passes through."""

    func: Callable[[Any], Awaitable[Any]]
    map_batch_size: int


# ---------------------------------------------------------------------------
# Result-stream terminals (result stream -> plain stream)
# ---------------------------------------------------------------------------


@dataclass
class FilterOk(Operator):
    """Drop all ``Err`` values silently and unwrap the ``Ok`` values."""


@dataclass
class OnError(Operator):
    """Call ``handler`` for each ``Err``, drop it, and unwrap ``Ok`` values."""

    handler: Callable[[Err], None]


# ---------------------------------------------------------------------------
# Terminal
# ---------------------------------------------------------------------------


@dataclass
class SinkOp(Operator):
    """Terminal node: write every element to ``sink``."""

    sink: Sink[Any]
