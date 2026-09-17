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
"""Lazily-evaluated dataflow pipelines as an embedded DSL.

Quick start::

    from neo4j_graphrag.pipeline import Pipeline

    results = (
        Pipeline.from_source(my_source)
        .map(transform)
        .flat_map(expand)
        .map_async_chunked(async_transform)
        .reduce(zero=identity, combine=merge)
        .collect()
    )

A pipeline built with :class:`Pipeline` is pure data — an operator graph.
Nothing executes until the definition is evaluated by an
:class:`Interpreter` (via :meth:`Pipeline.collect`,
:meth:`Pipeline.to_sink`, iteration, or an explicit interpreter).

.. note::
    This is not :class:`neo4j_graphrag.experimental.pipeline.Pipeline`.
    That class is a task-graph orchestrator built from ``Component``
    instances, wired with ``add_component`` / ``connect`` and run with
    ``await pipeline.run(...)``; it still backs ``SimpleKGPipeline`` and is
    scheduled for removal in 2.0.  This :class:`Pipeline` is an unrelated
    dataflow DSL over a stream of elements.  The two are never used
    together, and neither is re-exported from the other's package, so an
    import always names which one you mean.
"""

from neo4j_graphrag.pipeline.interpreter import Interpreter, LocalInterpreter
from neo4j_graphrag.pipeline.pipeline import Pipeline, ResultPipeline
from neo4j_graphrag.pipeline.result import Err, Ok, Result
from neo4j_graphrag.pipeline.sink import Sink
from neo4j_graphrag.pipeline.source import Source

__all__ = [
    "Interpreter",
    "LocalInterpreter",
    "Pipeline",
    "ResultPipeline",
    "Err",
    "Ok",
    "Result",
    "Sink",
    "Source",
]
