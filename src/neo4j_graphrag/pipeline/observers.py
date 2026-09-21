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
"""Stage observers: hooks for watching items flow through a pipeline.

Attach one or more :class:`StageObserver` instances to
:class:`~neo4j_graphrag.pipeline.interpreter.LocalInterpreter` to add
logging, metrics, tracing, or debugging without changing the pipeline
definition::

    from neo4j_graphrag.pipeline import LocalInterpreter, LoggingStageObserver

    stream = LocalInterpreter(observers=[LoggingStageObserver()]).evaluate(pipeline)

The interpreter wraps every operator's output stream with
:meth:`StageObserver.wrap`, which invokes the hooks around each item.
Observation is lazy: no hook runs until the stream is consumed, and a
partially consumed stream triggers hooks only for the items seen.

Hooks see items, never modify them — see :class:`StageObserver` for the
read-only contract they are expected to honour.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any, Generic, TypeVar

from neo4j_graphrag.pipeline.operators import Operator

__all__ = ["StageObserver", "LoggingStageObserver"]

logger = logging.getLogger(__name__)

#: Characters of an item's ``repr`` that :class:`LoggingStageObserver` logs
#: before clipping.
DEFAULT_MAX_REPR = 200

_T = TypeVar("_T")


class _ClippedRepr:
    """Lazily renders ``repr(item)``, clipped to *limit* characters.

    Passed to the logger instead of the item itself so that the ``repr`` of
    a large payload — an embedded chunk, a batch from ``grouped`` — is only
    built if the record is actually emitted, and never in full.
    """

    __slots__ = ("_item", "_limit")

    def __init__(self, item: Any, limit: int | None) -> None:
        self._item = item
        self._limit = limit

    def __repr__(self) -> str:
        text = repr(self._item)
        if self._limit is not None and len(text) > self._limit:
            return f"{text[: self._limit]}… [{len(text)} chars]"
        return text


class StageObserver(Generic[_T]):
    """Observes items as they leave each stage of a pipeline.

    Subclass and override whichever hooks you need; the defaults are
    no-ops.  The interpreter calls :meth:`wrap` once per operator; the
    default implementation drives the hooks as follows:

    * :meth:`before` — *op* has produced *item* and is about to yield it
      downstream.  The elapsed time since the previous item (or since the
      stage started) is the time *op* spent computing this item.
    * :meth:`after` — downstream has requested the next item, meaning
      *item* has been fully processed by every later stage.  The time
      between ``before`` and ``after`` measures downstream cost.
    * :meth:`on_error` — *op* failed.  Called by the interpreter rather
      than by :meth:`wrap`, so that a failure is reported once, against
      the stage it came from, instead of once per stage it propagates
      through.  Both kinds of failure arrive here:

      - **Captured**: a ``Try*`` stage (e.g. :meth:`Pipeline.map_safe`)
        turned a per-item exception into an
        :class:`~neo4j_graphrag.pipeline.result.Err`.  The stream carries
        on, and that same ``Err`` also reaches ``before``/``after`` as an
        ordinary item — which is where the failing value is visible.  The
        hook fires once, at the stage that captured the exception; an
        ``Err`` passing through later stages is not reported again.
      - **Fatal**: *op* raised without capture.  *error* propagates once
        this hook returns, and the stream is over.

    Because the hooks sit *between* stages, an observer attached to a
    pipeline of N operators sees each item N times — once per stage
    boundary — keyed by :attr:`Operator.name`.  A sink is the exception in
    one respect only: because it emits nothing of its own, the interpreter
    observes the writes instead — every item the sink writes is reported by
    ``before``/``after``, and a ``sink.write`` that raises reaches
    :meth:`on_error`.

    .. important::
        Hooks receive the **live item**, not a copy — the same object that
        continues downstream.  Treat it as read-only: mutating it changes
        what later stages see, and retaining it past the hook keeps it
        alive, defeating the streaming evaluation the DSL exists for.
        Derive what you need (a length, a type, an id) and let the item go.
    """

    def before(self, op: Operator, item: _T) -> None:
        """Called when *op* has produced *item*, before it flows downstream."""

    def after(self, op: Operator, item: _T) -> None:
        """Called when *item* has been consumed by every downstream stage."""

    def on_error(self, op: Operator, error: Exception) -> None:
        """Called when *op* fails with *error*.

        Either because a ``Try*`` stage captured it as an ``Err`` and the
        stream continues, or because it is about to propagate and end the
        stream — see the class docstring.
        """

    def wrap(self, op: Operator, stream: Iterator[_T]) -> Iterator[_T]:
        """Wrap *op*'s output stream, invoking the hooks around each item.

        Failures are not this method's concern — an exception here is one
        merely passing through, not a failure of this stage.  The
        interpreter reports failures to :meth:`on_error` instead, against
        the stage they came from, so an override of this method does not
        have to handle them.
        """
        for item in stream:
            self.before(op, item)
            yield item
            self.after(op, item)


class LoggingStageObserver(StageObserver[Any]):
    """Log stage boundaries, item flow, and failures.

    * Stage start and finish (with the emitted item count) at ``INFO``.
    * Every emitted item at ``DEBUG``.
    * Stage failures at ``ERROR`` — including per-item failures a ``Try*``
      stage captures as an ``Err``, so a partial-failure pipeline logs one
      ``ERROR`` per failed item.

    A stage whose stream is abandoned early (e.g. downstream of
    ``take``) never logs "finished".

    Args:
        log: Logger to write to.  Defaults to this module's logger,
            ``neo4j_graphrag.pipeline.observers``.
        max_repr: Characters of each item's ``repr`` to log at ``DEBUG``
            before clipping, so that one large payload cannot flood the
            log.  ``None`` logs the full ``repr``.
    """

    def __init__(
        self,
        log: logging.Logger | None = None,
        max_repr: int | None = DEFAULT_MAX_REPR,
    ) -> None:
        self._log = log if log is not None else logger
        self._max_repr = max_repr

    def wrap(self, op: Operator, stream: Iterator[Any]) -> Iterator[Any]:
        self._log.info("Stage %s: starting", op.name)
        count = 0
        for item in super().wrap(op, stream):
            count += 1
            yield item
        self._log.info("Stage %s: finished, %d item(s)", op.name, count)

    def before(self, op: Operator, item: Any) -> None:
        self._log.debug(
            "Stage %s: emitting %r", op.name, _ClippedRepr(item, self._max_repr)
        )

    def after(self, op: Operator, item: Any) -> None:
        self._log.debug(
            "Stage %s: consumed %r", op.name, _ClippedRepr(item, self._max_repr)
        )

    def on_error(self, op: Operator, error: Exception) -> None:
        self._log.error("Stage %s: failed with %r", op.name, error)
