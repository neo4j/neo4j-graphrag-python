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
"""Reads a completed batch job's prediction output lines back into plain text.

Counterpart to :mod:`~neo4j_graphrag.llm.batch.formatter`: a
:class:`~neo4j_graphrag.llm.batch.formatter.BatchRequestFormatter` builds the
request line a client submits; a :class:`BaseBatchResponseReader` parses the
prediction line the job writes back, once
:meth:`~neo4j_graphrag.llm.batch.client.BaseBatchClient.wait_for` reports the
job finished. Kept separate from both for the same reason they are separate
from each other: the response envelope is provider-specific and has nothing
to do with submitting/polling the job or building its input.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BatchResponseRecord:
    """One parsed prediction, correlated back to the request that produced it.

    Attributes:
        key: The same correlation id the request line carried (see
            :class:`~neo4j_graphrag.llm.batch.formatter.BatchRequestLine`).
        content: The model's text output, or ``None`` when *error* is set.
        error: The provider's reason this row failed, or ``None`` on success.
            A batch job can partially fail — some rows succeed while others
            error — so this is per-row rather than per-job (contrast
            :meth:`~neo4j_graphrag.llm.batch.client.BaseBatchClient.job_failure_reason`,
            which reports why the whole job failed).
    """

    key: str
    content: str | None
    error: str | None = None


class BaseBatchResponseReader(ABC):
    """Parses one line of a completed batch job's prediction output file.

    Subclasses implement the provider-specific response envelope.
    """

    @abstractmethod
    def read(self, line: str) -> BatchResponseRecord:
        """Parse one JSONL *line* from the prediction output file."""


# ---------------------------------------------------------------------------
# Vertex AI implementation
# ---------------------------------------------------------------------------


class VertexBatchResponseReader(BaseBatchResponseReader):
    """Reads Vertex AI Gemini batch prediction output lines.

    Reads the shape a :class:`~neo4j_graphrag.llm.batch.formatter.VertexBatchRequestFormatter`
    request comes back as::

        {
          "key": "<caller-supplied id>",
          "status": "",
          "response": {
            "candidates": [
              {"content": {"parts": [{"text": "..."}], "role": "model"}}
            ],
            "usageMetadata": {...}
          }
        }

    A row that failed has a non-empty ``status`` and no ``response``.
    """

    def read(self, line: str) -> BatchResponseRecord:
        record = json.loads(line)
        key = record["key"]
        status = record.get("status") or None
        response = record.get("response")
        if status or response is None:
            return BatchResponseRecord(
                key=key, content=None, error=status or "missing response"
            )
        candidates: list[dict[str, Any]] = response.get("candidates") or []
        if not candidates:
            return BatchResponseRecord(
                key=key,
                content=None,
                error="no candidates returned (generation may have been blocked)",
            )
        return BatchResponseRecord(
            key=key, content=self._content(candidates), error=None
        )

    @staticmethod
    def _content(candidates: list[dict[str, Any]]) -> str:
        parts: list[dict[str, Any]] = (
            candidates[0].get("content", {}).get("parts") or []
        )
        return "\n".join(str(part["text"]) for part in parts if part.get("text"))
