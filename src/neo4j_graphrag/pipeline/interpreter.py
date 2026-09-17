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
"""Interpreters evaluate pipeline definitions.

A pipeline built with :class:`~neo4j_graphrag.pipeline.pipeline.Pipeline`
is pure data — a linked list of
:class:`~neo4j_graphrag.pipeline.operators.Operator` nodes.  An
:class:`Interpreter` walks that list and executes it.

:class:`LocalInterpreter` evaluates the pipeline in the current process
using lazy generators: nothing executes until the returned stream is
consumed (``collect()``, ``to_sink()``, or direct iteration).

Async stages are evaluated **blocking**: each chunk of ``map_batch_size``
items is dispatched with ``asyncio.gather`` inside its own
``asyncio.run()`` call, and its results are yielded before the next chunk
is fetched.  This bounds memory usage and keeps the interpreter
synchronous, at the cost of two restrictions:

* async operators must not be evaluated from within a running event loop
  (use ``asyncio.to_thread`` at the call site if needed), and
* async clients that bind to an event loop (``httpx.AsyncClient``,
  ``aiohttp.ClientSession``) must be created *inside* the async function
  rather than shared across chunks, because each chunk runs on a fresh
  event loop.

A future ``AsyncInterpreter`` (whole chain on a single event loop) would
lift both restrictions; the operator-graph representation already supports
it.
"""

from __future__ import annotations

import asyncio
import itertools
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Coroutine, Iterable, Iterator
from functools import reduce as _reduce
from typing import TYPE_CHECKING, Any, TypeVar

from neo4j_graphrag.pipeline.operators import (
    Filter,
    FilterOk,
    FlattenOk,
    FlatMap,
    Grouped,
    Map,
    MapAsyncChunked,
    MapOk,
    OnError,
    ReduceOp,
    SinkOp,
    Skip,
    SourceOp,
    Take,
    TakeWhile,
    TryMap,
    TryMapAsyncChunked,
    TryMapOk,
    TryMapOkAsyncChunked,
)
from neo4j_graphrag.pipeline.result import Err, Ok
from neo4j_graphrag.pipeline.sink import Sink
from neo4j_graphrag.pipeline.source import Source

if TYPE_CHECKING:
    from neo4j_graphrag.pipeline.pipeline import Pipeline, ResultPipeline

__all__ = ["Interpreter", "LocalInterpreter"]

_T = TypeVar("_T")
_U = TypeVar("_U")


class Interpreter(ABC):
    """Base class for pipeline interpreters.

    An interpreter executes a pipeline definition and returns the resulting
    stream.  Whether evaluation is lazy or eager, local or distributed, is
    up to the implementation.
    """

    @abstractmethod
    def evaluate(self, pipeline: Pipeline[Any] | ResultPipeline[Any]) -> Iterator[Any]:
        """Evaluate *pipeline* and return its output stream.

        Implementations should not execute any user code before the
        returned stream is consumed.  A pipeline ending in a sink returns a
        stream that yields nothing but writes to the sink as it is drained,
        so callers must exhaust it — :meth:`Pipeline.to_sink` does this.
        """


# ---------------------------------------------------------------------------
# Evaluation helpers (shared generator building blocks)
# ---------------------------------------------------------------------------


def _batched(iterable: Iterable[_T], n: int) -> Iterator[list[_T]]:
    """Yield successive non-overlapping lists of length *n* from *iterable*."""
    it = iter(iterable)
    while batch := list(itertools.islice(it, n)):
        yield batch


def _iter_chunked_async(
    stream: Iterable[_T],
    map_batch_size: int,
    make_chunk_coro: Callable[[list[_T]], Coroutine[None, None, list[_U]]],
) -> Iterator[_U]:
    """Shared engine for chunked async operators.

    Iterates *stream* in chunks of *map_batch_size* and, for each chunk,
    calls ``asyncio.run(make_chunk_coro(chunk))``, yielding all results
    before moving to the next chunk.

    Raises:
        ValueError: If *map_batch_size* < 1.  (Builders validate eagerly;
            this is a backstop for hand-built operator graphs.)
    """
    if map_batch_size < 1:
        raise ValueError(f"map_batch_size must be >= 1, got {map_batch_size!r}")
    for chunk in _batched(stream, map_batch_size):
        yield from asyncio.run(make_chunk_coro(chunk))


def _make_map_chunk_coro(
    func: Callable[[_T], Awaitable[_U]],
) -> Callable[[list[_T]], Coroutine[None, None, list[_U]]]:
    async def _run_chunk(chunk: list[_T]) -> list[_U]:
        return list(await asyncio.gather(*[func(item) for item in chunk]))

    return _run_chunk


def _make_try_chunk_coro(
    func: Callable[[_T], Awaitable[_U]],
) -> Callable[[list[_T]], Coroutine[None, None, list[Ok[_U] | Err]]]:
    async def _run_chunk(chunk: list[_T]) -> list[Ok[_U] | Err]:
        raw = await asyncio.gather(
            *[func(item) for item in chunk], return_exceptions=True
        )
        results: list[Ok[_U] | Err] = []
        for r in raw:
            if isinstance(r, Exception):
                results.append(Err(exception=r))
            elif isinstance(r, BaseException):
                # SystemExit, KeyboardInterrupt, CancelledError: fatal, re-raise.
                raise r
            else:
                results.append(Ok(value=r))
        return results

    return _run_chunk


def _make_try_ok_chunk_coro(
    func: Callable[[Any], Awaitable[Any]],
) -> Callable[[list[Any]], Coroutine[None, None, list[Any]]]:
    async def _process(item: Ok[Any] | Err) -> Ok[Any] | Err:
        if isinstance(item, Err):
            return item
        try:
            return Ok(value=await func(item.value))
        except Exception as e:
            return Err(exception=e)

    async def _run_chunk(chunk: list[Any]) -> list[Any]:
        return list(await asyncio.gather(*[_process(item) for item in chunk]))

    return _run_chunk


def _try_map(stream: Iterator[Any], func: Callable[[Any], Any]) -> Iterator[Any]:
    for item in stream:
        try:
            yield Ok(value=func(item))
        except Exception as e:
            yield Err(exception=e)


def _map_ok(stream: Iterator[Any], func: Callable[[Any], Any]) -> Iterator[Any]:
    for item in stream:
        if isinstance(item, Err):
            yield item
        else:
            yield Ok(value=func(item.value))


def _try_map_ok(stream: Iterator[Any], func: Callable[[Any], Any]) -> Iterator[Any]:
    for item in stream:
        if isinstance(item, Err):
            yield item
        else:
            try:
                yield Ok(value=func(item.value))
            except Exception as e:
                yield Err(exception=e)


def _read_source(source: Source[Any]) -> Iterator[Any]:
    """Read *source*, deferred until the stream is consumed.

    ``read()`` may open a file or a connection, so it must not run while
    the generator chain is merely being built.
    """
    yield from source.read()


def _reduce_lazy(
    stream: Iterator[Any], zero: Any, combine: Callable[[Any, Any], Any]
) -> Iterator[Any]:
    """Fold *stream* into a single element, deferred until consumed.

    The fold itself is not incremental — it drains the upstream — but it
    does so on first ``next()`` rather than when the chain is built, so
    ``evaluate()`` stays free of side effects.
    """
    yield _reduce(combine, stream, zero)


def _to_sink(stream: Iterator[Any], sink: Sink[Any]) -> Iterator[Any]:
    """Write every element of *stream* to *sink*, yielding nothing."""
    for item in stream:
        sink.write(item)
    yield from ()


def _flatten_ok(stream: Iterator[Any]) -> Iterator[Any]:
    for item in stream:
        if isinstance(item, Err):
            yield item
        else:
            yield from (Ok(value=v) for v in item.value)


def _on_error(stream: Iterator[Any], handler: Callable[[Err], None]) -> Iterator[Any]:
    for item in stream:
        if isinstance(item, Err):
            handler(item)
        else:
            yield item.value


# ---------------------------------------------------------------------------
# Local interpreter
# ---------------------------------------------------------------------------


class LocalInterpreter(Interpreter):
    """Evaluate a pipeline in the current process with lazy generators.

    Folds the operator list into a single generator chain.  Building the
    chain runs no user code: every operator — including ``ReduceOp`` and
    ``SinkOp``, which both drain their upstream — defers its work until the
    returned stream is consumed.  The chunked-async operators are the one
    exception to incremental evaluation: each chunk blocks in
    ``asyncio.run`` while it is being produced.
    """

    def evaluate(self, pipeline: Pipeline[Any] | ResultPipeline[Any]) -> Iterator[Any]:
        stream: Iterator[Any] = iter(())
        for op in pipeline.pipeline_operators:
            match op:
                case SourceOp(source=source):
                    stream = _read_source(source)
                case Map(func=func):
                    stream = map(func, stream)
                case FlatMap(func=func):
                    stream = itertools.chain.from_iterable(map(func, stream))
                case Filter(predicate=predicate):
                    stream = filter(predicate, stream)
                case Take(n=n):
                    stream = itertools.islice(stream, n)
                case TakeWhile(predicate=predicate):
                    stream = itertools.takewhile(predicate, stream)
                case Skip(n=n):
                    stream = itertools.islice(stream, n, None)
                case Grouped(size=size):
                    stream = _batched(stream, size)
                case ReduceOp(zero=zero, combine=combine):
                    stream = _reduce_lazy(stream, zero, combine)
                case MapAsyncChunked(func=func, map_batch_size=batch_size):
                    stream = _iter_chunked_async(
                        stream, batch_size, _make_map_chunk_coro(func)
                    )
                case TryMap(func=func):
                    stream = _try_map(stream, func)
                case TryMapAsyncChunked(func=func, map_batch_size=batch_size):
                    stream = _iter_chunked_async(
                        stream, batch_size, _make_try_chunk_coro(func)
                    )
                case MapOk(func=func):
                    stream = _map_ok(stream, func)
                case FlattenOk():
                    stream = _flatten_ok(stream)
                case TryMapOk(func=func):
                    stream = _try_map_ok(stream, func)
                case TryMapOkAsyncChunked(func=func, map_batch_size=batch_size):
                    stream = _iter_chunked_async(
                        stream, batch_size, _make_try_ok_chunk_coro(func)
                    )
                case FilterOk():
                    stream = (item.value for item in stream if isinstance(item, Ok))
                case OnError(handler=handler):
                    stream = _on_error(stream, handler)
                case SinkOp(sink=sink):
                    stream = _to_sink(stream, sink)
                case _:  # pragma: no cover
                    raise TypeError(f"Unknown operator: {op!r}")
        return stream
