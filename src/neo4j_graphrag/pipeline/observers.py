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
"""

from __future__ import annotations

import logging
from abc import ABC
from collections.abc import Iterator
from typing import Any, Generic, TypeVar

from neo4j_graphrag.pipeline.operators import Operator

__all__ = ["StageObserver", "LoggingStageObserver"]

logger = logging.getLogger(__name__)

_T = TypeVar("_T")


class StageObserver(ABC, Generic[_T]):
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
    * :meth:`on_error` — *op* (or a stage upstream of it) raised;
      *error* propagates once this hook returns.  The item that caused
      the failure is not visible at this level — to observe per-item
      failures, use a ``Try*`` stage (e.g. :meth:`Pipeline.map_safe`) and
      watch the ``Err`` values flow through ``before``/``after``.

    Because the hooks sit *between* stages, an observer attached to a
    pipeline of N operators sees each item N times — once per stage
    boundary — keyed by :attr:`Operator.name`.
    """

    def before(self, op: Operator, item: _T) -> None:
        """Called when *op* has produced *item*, before it flows downstream."""

    def after(self, op: Operator, item: _T) -> None:
        """Called when *item* has been consumed by every downstream stage."""

    def on_error(self, op: Operator, error: Exception) -> None:
        """Called when evaluating *op* raises *error* (which then propagates)."""

    def wrap(self, op: Operator, stream: Iterator[_T]) -> Iterator[_T]:
        """Wrap *op*'s output stream, invoking the hooks around each item."""
        iterator = iter(stream)
        while True:
            try:
                item = next(iterator)
            except StopIteration:
                return
            except Exception as e:
                self.on_error(op, e)
                raise
            self.before(op, item)
            yield item
            self.after(op, item)


class LoggingStageObserver(StageObserver[Any]):
    """Log stage boundaries, item flow, and failures.

    * Stage start and finish (with the emitted item count) at ``INFO``.
    * Every emitted item at ``DEBUG``.
    * Stage failures at ``ERROR``.

    A stage whose stream is abandoned early (e.g. downstream of
    ``take``) never logs "finished".
    """

    def __init__(self, log: logging.Logger | None = None) -> None:
        self._log = log if log is not None else logger

    def wrap(self, op: Operator, stream: Iterator[Any]) -> Iterator[Any]:
        self._log.info("Stage %s: starting", op.name)
        count = 0
        for item in super().wrap(op, stream):
            count += 1
            yield item
        self._log.info("Stage %s: finished, %d item(s)", op.name, count)

    def before(self, op: Operator, item: Any) -> None:
        self._log.debug("Stage %s: emitting %r", op.name, item)

    def on_error(self, op: Operator, error: Exception) -> None:
        self._log.error("Stage %s: failed with %r", op.name, error)
