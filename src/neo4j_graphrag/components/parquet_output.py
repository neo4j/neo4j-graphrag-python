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
"""Protocol for Parquet output destinations (local or cloud)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ParquetOutputDestination(Protocol):
    """Protocol for writing Parquet file bytes to a destination (local or cloud).

    Implementations might write to a local directory, GCS, S3, Azure Blob, etc.
    GraphRAG does not ship implementations; consumers provide their own.
    """

    @property
    def output_path(self) -> str:
        """Logical output path/prefix for this destination.

        Used for collision handling and metadata. Format is backend-specific
        (e.g. a directory path, or a bucket prefix like 'gs://bucket/nodes/').
        """
        ...

    async def write(self, data: bytes, filename: str) -> None:
        """Write bytes to the destination under the given filename.

        Args:
            data: Parquet file content.
            filename: Filename or key (e.g. 'Person.parquet'). No path separators;
                the destination appends this to its output_path as appropriate.
        """
        ...
