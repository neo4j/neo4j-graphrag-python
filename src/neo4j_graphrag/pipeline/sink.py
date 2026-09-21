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
"""Sink abstract base class for the pipeline DSL.

A ``Sink`` is the terminal destination for pipeline output (e.g. a file or
a Neo4j database).  Implementations must subclass :class:`Sink` and
implement :meth:`Sink.write`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

__all__ = ["Sink"]

T_contra = TypeVar("T_contra", contravariant=True)


class Sink(ABC, Generic[T_contra]):
    """Abstract base class for objects that can receive pipeline elements.

    Subclasses must implement :meth:`write`, which accepts a single
    element of type ``T_contra`` and persists it to the backing store.

    Failures are fatal by design: an exception raised by ``write`` is not
    captured as an :class:`~neo4j_graphrag.pipeline.result.Err`, it
    propagates out of :meth:`~neo4j_graphrag.pipeline.pipeline.Pipeline.to_sink`
    and aborts the run.  Elements written before the failure stay written —
    a sink is not transactional unless the implementation makes it so.  A
    sink that should tolerate per-element failures must catch them itself,
    e.g. by recording them and returning normally.

    Example::

        class InMemorySink(Sink[Any]):
            def __init__(self) -> None:
                self.received: list[Any] = []

            def write(self, element: Any) -> None:
                self.received.append(element)
    """

    @abstractmethod
    def write(self, element: T_contra) -> None:
        """Write a single *element* to the backing store."""
        ...
