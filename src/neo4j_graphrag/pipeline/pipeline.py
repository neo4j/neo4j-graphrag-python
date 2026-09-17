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
"""Embedded pipeline DSL: fluent builders over a pure-data operator graph.

.. note::
    This is not :class:`neo4j_graphrag.experimental.pipeline.Pipeline`.
    That class is a task-graph orchestrator built from ``Component``
    instances, wired with ``add_component`` / ``connect`` and run with
    ``await pipeline.run(...)``; it still backs ``SimpleKGPipeline`` and is
    scheduled for removal in 2.0.  This :class:`Pipeline` is an unrelated
    dataflow DSL over a stream of elements.  The two are never used
    together, and neither is re-exported from the other's package, so an
    import always names which one you mean.

:class:`Pipeline` and :class:`ResultPipeline` are **builders**.  Every
operator method appends an
:class:`~neo4j_graphrag.pipeline.operators.Operator` node to the definition
and returns a new builder — nothing executes.  Evaluation is deferred to an
:class:`~neo4j_graphrag.pipeline.interpreter.Interpreter`; the terminal
methods :meth:`Pipeline.collect` and :meth:`Pipeline.to_sink` (and direct
iteration) run the pipeline with the default
:class:`~neo4j_graphrag.pipeline.interpreter.LocalInterpreter`.

Because the definition is data, it can also be inspected, validated, or
rendered without executing it, and alternative interpreters (async,
multiprocessing, …) can evaluate the same definition differently.

Method names compose from three independent suffixes:

``_safe``
    Capture per-item exceptions as :class:`~neo4j_graphrag.pipeline.result.Err`
    instead of aborting the stream.  Returns a :class:`ResultPipeline`.
``_ok``
    Operate on the value inside each ``Ok``, passing ``Err`` through
    untouched.  Only available on :class:`ResultPipeline`.
``_async_chunked``
    Apply an async function to ``map_batch_size`` items concurrently.

1-to-many stages are *not* a fourth suffix: use a ``map`` variant followed
by :meth:`Pipeline.flat_map` or :meth:`ResultPipeline.flatten_ok`.

Example — partial-failure safe pipeline::

    from neo4j_graphrag.pipeline import Err, Pipeline

    errors: list[Err] = []
    results = (
        Pipeline.from_source(source)
        .map_async_chunked_safe(load)      # ResultPipeline[Bytes]
        .map_ok(parse)                     # ResultPipeline[Doc]
        .map_ok(split).flatten_ok()        # ResultPipeline[Chunk]
        .map_async_chunked_safe(embed)     # ResultPipeline[EmbeddedChunk]
        .on_error(errors.append)           # Pipeline[EmbeddedChunk]
        .collect()
    )
"""

from __future__ import annotations

from collections import deque
from collections.abc import Awaitable, Callable, Iterable, Iterator
from typing import Any, Generic, TypeVar

from neo4j_graphrag.pipeline import operators as ops
from neo4j_graphrag.pipeline.interpreter import LocalInterpreter
from neo4j_graphrag.pipeline.result import Err, Ok
from neo4j_graphrag.pipeline.sink import Sink
from neo4j_graphrag.pipeline.source import Source

__all__ = ["Pipeline", "ResultPipeline"]

T = TypeVar("T")
S = TypeVar("S")
U = TypeVar("U")
A = TypeVar("A")
ResI = TypeVar("ResI", covariant=True)
OutO = TypeVar("OutO")


def _validate_batch_size(map_batch_size: int) -> None:
    if map_batch_size < 1:
        raise ValueError(f"map_batch_size must be >= 1, got {map_batch_size!r}")


class _IterableSource(Source[Any]):
    """Adapter wrapping any ``Iterable`` as a :class:`Source`."""

    def __init__(self, iterable: Iterable[Any]) -> None:
        self._iterable = iterable

    def read(self) -> Iterable[Any]:
        return self._iterable


class Pipeline(Generic[T]):
    """A lazily-evaluated pipeline definition.

    Construct from any ``Iterable`` or with :meth:`from_source`::

        Pipeline([1, 2, 3]).map(lambda x: x * 2).collect()  # [2, 4, 6]

    Each operator appends a node to the operator graph and returns a fresh
    ``Pipeline`` — nothing runs until the pipeline is consumed via
    :meth:`collect`, :meth:`to_sink`, iteration, or an explicit
    :class:`~neo4j_graphrag.pipeline.interpreter.Interpreter`.
    """

    def __init__(self, stream: Iterable[T]) -> None:
        self._tail: ops.Operator = ops.SourceOp(
            prev=None, source=_IterableSource(stream)
        )

    @classmethod
    def from_source(cls, source: Source[S]) -> Pipeline[S]:
        """Create a pipeline whose elements come from *source*.

        Args:
            source: A :class:`~neo4j_graphrag.pipeline.source.Source`
                implementation.

        Returns:
            A new :class:`Pipeline` typed to the element type of *source*.
        """
        return cls._wrap(ops.SourceOp(prev=None, source=source))

    @staticmethod
    def _wrap(tail: ops.Operator) -> Pipeline[Any]:
        """Build a builder instance around an existing tail node."""
        pipe = Pipeline.__new__(Pipeline)
        pipe._tail = tail
        return pipe

    @property
    def pipeline_operators(self) -> list[ops.Operator]:
        """The operator chain in evaluation order (source first)."""
        operators: list[ops.Operator] = []
        current: ops.Operator | None = self._tail
        while current is not None:
            operators.append(current)
            current = current.prev
        return operators[::-1]

    # ------------------------------------------------------------------
    # Synchronous operators
    # ------------------------------------------------------------------

    def map(self, func: Callable[[T], U]) -> Pipeline[U]:
        """Apply *func* to every element (1-to-1).

        Args:
            func: A callable ``T → U``.
        """
        return self._wrap(ops.Map(prev=self._tail, func=func))

    def flat_map(self, func: Callable[[T], Iterable[U]]) -> Pipeline[U]:
        """Apply *func* and flatten one level (1-to-many).

        Args:
            func: A callable ``T → Iterable[U]``.
        """
        return self._wrap(ops.FlatMap(prev=self._tail, func=func))

    def take(self, n: int) -> Pipeline[T]:
        """Take the first *n* elements from the stream.

        Raises:
            ValueError: If *n* < 0.
        """
        if n < 0:
            raise ValueError(f"n must be >= 0, got {n!r}")
        return self._wrap(ops.Take(prev=self._tail, n=n))

    def take_while(self, predicate: Callable[[T], bool]) -> Pipeline[T]:
        """Take elements while *predicate* is ``True``.

        Stops consuming the stream as soon as *predicate* returns ``False``
        for the first time — unlike :meth:`filter`, which skips non-matching
        elements and continues.
        """
        return self._wrap(ops.TakeWhile(prev=self._tail, predicate=predicate))

    def skip(self, n: int) -> Pipeline[T]:
        """Skip the first *n* elements from the stream.

        Raises:
            ValueError: If *n* < 0.
        """
        if n < 0:
            raise ValueError(f"n must be >= 0, got {n!r}")
        return self._wrap(ops.Skip(prev=self._tail, n=n))

    def filter(self, predicate: Callable[[T], bool]) -> Pipeline[T]:
        """Keep only elements for which *predicate* is ``True``."""
        return self._wrap(ops.Filter(prev=self._tail, predicate=predicate))

    def grouped(self, size: int) -> Pipeline[list[T]]:
        """Collect elements into batches of up to *size* items.

        Raises:
            ValueError: If *size* < 1.
        """
        if size < 1:
            raise ValueError(f"grouped size must be >= 1, got {size!r}")
        return self._wrap(ops.Grouped(prev=self._tail, size=size))

    def reduce(self, zero: A, combine: Callable[[A, T], A]) -> Pipeline[A]:
        """Fold all elements into a single value, emitted as a one-element stream.

        The *zero* / *combine* pair should form a Monoid: *combine* must be
        associative and *zero* its identity element.

        Args:
            zero: The identity / initial accumulator value.
            combine: An associative binary callable ``(A, T) → A``.
        """
        return self._wrap(ops.ReduceOp(prev=self._tail, zero=zero, combine=combine))

    # ------------------------------------------------------------------
    # Async operators (blocking under LocalInterpreter)
    # ------------------------------------------------------------------

    def map_async_chunked(
        self,
        func: Callable[[T], Awaitable[U]],
        map_batch_size: int = 100,
    ) -> Pipeline[U]:
        """Apply async *func* concurrently to elements in chunks.

        Under the :class:`~neo4j_graphrag.pipeline.interpreter.LocalInterpreter`,
        *map_batch_size* elements at a time are dispatched via
        ``asyncio.gather``; each chunk's results are yielded before the next
        chunk is fetched, bounding memory usage.  Evaluation is **blocking**
        (``asyncio.run`` per chunk) — see the interpreter documentation for
        the restrictions this implies.

        Args:
            func: An async callable ``T → Awaitable[U]``.
            map_batch_size: Items to process concurrently per chunk.
                Must be >= 1.

        Raises:
            ValueError: If *map_batch_size* < 1.
        """
        _validate_batch_size(map_batch_size)
        return self._wrap(
            ops.MapAsyncChunked(
                prev=self._tail, func=func, map_batch_size=map_batch_size
            )
        )

    # ------------------------------------------------------------------
    # Partial-failure safe operators (Pipeline -> ResultPipeline)
    # ------------------------------------------------------------------

    def map_safe(self, func: Callable[[T], U]) -> ResultPipeline[U]:
        """Apply *func* to every element, capturing exceptions as ``Err``.

        Each item is individually wrapped in
        :class:`~neo4j_graphrag.pipeline.result.Ok` on success or
        :class:`~neo4j_graphrag.pipeline.result.Err` on failure.  Unlike
        :meth:`map`, an exception raised by *func* does not abort the stream.
        """
        return ResultPipeline._wrap(ops.TryMap(prev=self._tail, func=func))

    def map_async_chunked_safe(
        self,
        func: Callable[[T], Awaitable[U]],
        map_batch_size: int = 100,
    ) -> ResultPipeline[U]:
        """Like :meth:`map_async_chunked` but captures per-item exceptions as ``Err``.

        Within each chunk, ``asyncio.gather`` is called with
        ``return_exceptions=True`` so one failing item does not abort the
        chunk.

        Raises:
            ValueError: If *map_batch_size* < 1.
        """
        _validate_batch_size(map_batch_size)
        return ResultPipeline._wrap(
            ops.TryMapAsyncChunked(
                prev=self._tail, func=func, map_batch_size=map_batch_size
            )
        )

    # ------------------------------------------------------------------
    # Terminal operators
    # ------------------------------------------------------------------

    def collect(self) -> list[T]:
        """Evaluate the pipeline with the default interpreter and
        materialise the stream into a list.

        Returns:
            All elements produced by the pipeline in order.
        """
        return list(LocalInterpreter().evaluate(self))

    def to_sink(self, sink: Sink[T]) -> None:
        """Evaluate the pipeline with the default interpreter, writing each
        element to *sink*.

        An exception raised by ``sink.write`` propagates and aborts the run;
        elements already written are not rolled back.  See
        :class:`~neo4j_graphrag.pipeline.sink.Sink` for the rationale.

        Args:
            sink: A :class:`~neo4j_graphrag.pipeline.sink.Sink`
                implementation.
        """
        stream = LocalInterpreter().evaluate(
            self._wrap(ops.SinkOp(prev=self._tail, sink=sink))
        )
        deque(stream, maxlen=0)  # drain: the writes happen as the stream is consumed

    def __iter__(self) -> Iterator[T]:
        return LocalInterpreter().evaluate(self)


class ResultPipeline(Generic[ResI]):
    """A pipeline whose elements are ``Ok[ResI] | Err`` values.

    Returned by the ``_safe`` operators on :class:`Pipeline`.  All
    Result-aware combinators pass
    :class:`~neo4j_graphrag.pipeline.result.Err` values through unchanged
    unless otherwise noted.

    Construct via the ``_safe`` operators::

        result_stream: ResultPipeline[int] = (
            Pipeline([1, 2, 3]).map_async_chunked_safe(f)
        )
        clean: Pipeline[int] = result_stream.on_error(counter.record)
    """

    def __init__(self, stream: Iterable[Ok[ResI] | Err]) -> None:
        self._tail: ops.Operator = ops.SourceOp(
            prev=None, source=_IterableSource(stream)
        )

    @staticmethod
    def _wrap(tail: ops.Operator) -> ResultPipeline[Any]:
        """Build a builder instance around an existing tail node."""
        pipe = ResultPipeline.__new__(ResultPipeline)
        pipe._tail = tail
        return pipe

    @property
    def pipeline_operators(self) -> list[ops.Operator]:
        """The operator chain in evaluation order (source first)."""
        operators: list[ops.Operator] = []
        current: ops.Operator | None = self._tail
        while current is not None:
            operators.append(current)
            current = current.prev
        return operators[::-1]

    # ------------------------------------------------------------------
    # Result-aware combinators
    # ------------------------------------------------------------------

    def map_ok(self, func: Callable[[ResI], OutO]) -> ResultPipeline[OutO]:
        """Apply *func* to the value inside each ``Ok``; ``Err`` passes through.

        Exceptions raised by *func* propagate and abort the stream — use
        :meth:`map_safe` to capture them as ``Err`` instead.
        """
        return self._wrap(ops.MapOk(prev=self._tail, func=func))

    def flatten_ok(self: ResultPipeline[Iterable[OutO]]) -> ResultPipeline[OutO]:
        """Expand each ``Ok`` holding an iterable into one ``Ok`` per item;
        ``Err`` passes through unchanged.

        This is how the DSL expresses 1-to-many stages on a result stream:
        pair it with whichever ``map`` variant has the error semantics you
        want, instead of a dedicated ``flat_map`` for each combination::

            .map_ok(split).flatten_ok()                 # split must not fail
            .map_safe(split).flatten_ok()               # capture failures
            .map_async_chunked_safe(split).flatten_ok()  # async, capture

        The expansion itself is not error-capturing: if a value turns out
        not to be iterable, or a lazy iterable raises while being consumed,
        that exception propagates.  Return a materialised sequence from the
        preceding ``map_safe`` if you need those failures as ``Err``.
        """
        return self._wrap(ops.FlattenOk(prev=self._tail))

    def map_safe(self, func: Callable[[ResI], OutO]) -> ResultPipeline[OutO]:
        """Apply *func* to each ``Ok`` value, capturing exceptions as ``Err``;
        existing ``Err`` values pass through unchanged.

        Unlike :meth:`map_ok`, an exception raised by *func* does not
        propagate — it is captured as a new ``Err`` so the stream continues
        processing remaining items.
        """
        return self._wrap(ops.TryMapOk(prev=self._tail, func=func))

    def map_async_chunked_safe(
        self,
        func: Callable[[ResI], Awaitable[OutO]],
        map_batch_size: int = 100,
    ) -> ResultPipeline[OutO]:
        """Apply async *func* to each ``Ok`` value in concurrent chunks;
        ``Err`` values pass through unchanged and new exceptions are
        captured as ``Err``.

        Raises:
            ValueError: If *map_batch_size* < 1.
        """
        _validate_batch_size(map_batch_size)
        return self._wrap(
            ops.TryMapOkAsyncChunked(
                prev=self._tail, func=func, map_batch_size=map_batch_size
            )
        )

    # ------------------------------------------------------------------
    # Result terminals
    # ------------------------------------------------------------------

    def filter_ok(self) -> Pipeline[ResI]:
        """Drop all ``Err`` values silently and unwrap the ``Ok`` values.

        No handler is called for dropped errors.  Use :meth:`on_error`
        instead when failures should be logged or counted.
        """
        return Pipeline._wrap(ops.FilterOk(prev=self._tail))

    def on_error(self, handler: Callable[[Err], None]) -> Pipeline[ResI]:
        """Call *handler* for each ``Err``, drop it from the stream, and
        unwrap ``Ok`` values.

        Returns a plain :class:`Pipeline` with no ``Result`` wrapper,
        suitable for passing to non-Result-aware operators or sinks.

        Args:
            handler: A callable ``Err → None`` invoked for every failed
                item, e.g. a logging callback or a failure counter.
        """
        return Pipeline._wrap(ops.OnError(prev=self._tail, handler=handler))

    def collect(self) -> list[Ok[ResI] | Err]:
        """Evaluate the pipeline and materialise the raw ``Ok``/``Err``
        stream into a list."""
        return list(LocalInterpreter().evaluate(self))

    def __iter__(self) -> Iterator[Ok[ResI] | Err]:
        return LocalInterpreter().evaluate(self)
