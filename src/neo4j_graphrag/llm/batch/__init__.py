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
"""Batch inference: one submit/poll interface, one formatter, one reader per provider.

* :mod:`~neo4j_graphrag.llm.batch.client` — :class:`BaseBatchClient` (the
  shared submit/poll interface and polling loop) plus one subclass per
  transport, e.g. :class:`VertexBatchClient` for Vertex AI via ``google.genai``.
* :mod:`~neo4j_graphrag.llm.batch.formatter` — :class:`BatchRequestFormatter`
  (builds the ``requests.jsonl`` lines a client's
  :meth:`~neo4j_graphrag.llm.batch.client.BaseBatchClient.submit` is handed)
  plus one subclass per provider's request envelope, e.g.
  :class:`VertexBatchRequestFormatter`; :class:`BatchModelParams` normalises
  generation parameters for each formatter (:class:`VertexModelParams` for
  Vertex).
* :mod:`~neo4j_graphrag.llm.batch.reader` — :class:`BaseBatchResponseReader`
  (parses the prediction lines a finished job writes back) plus one subclass
  per provider's response envelope, e.g. :class:`VertexBatchResponseReader`.
"""

from neo4j_graphrag.llm.batch.client import (
    BATCH_JOB_SUCCEEDED_STATE,
    DEFAULT_MAX_WAIT_SECONDS,
    TERMINAL_BATCH_JOB_STATES,
    BaseBatchClient,
    BatchJob,
    VertexBatchClient,
)
from neo4j_graphrag.llm.batch.formatter import (
    BatchModelParams,
    BatchRequestFormatter,
    BatchRequestLine,
    VertexBatchRequestFormatter,
    VertexModelParams,
)
from neo4j_graphrag.llm.batch.reader import (
    BaseBatchResponseReader,
    BatchResponseRecord,
    VertexBatchResponseReader,
)

__all__ = [
    "BATCH_JOB_SUCCEEDED_STATE",
    "DEFAULT_MAX_WAIT_SECONDS",
    "TERMINAL_BATCH_JOB_STATES",
    "BaseBatchClient",
    "BaseBatchResponseReader",
    "BatchJob",
    "BatchModelParams",
    "BatchRequestFormatter",
    "BatchRequestLine",
    "BatchResponseRecord",
    "VertexBatchClient",
    "VertexBatchRequestFormatter",
    "VertexBatchResponseReader",
    "VertexModelParams",
]
