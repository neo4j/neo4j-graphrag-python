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
"""Filename collision handler for Parquet file writing."""

from __future__ import annotations

from pathlib import Path
from typing import Union


class FilenameCollisionHandler:
    """Handles filename collisions by adding numeric suffixes.

    Tracks filename collisions per output path and generates unique filenames
    by appending _n suffixes when the same base filename is requested more
    than once for the same output path.

    Example:

    .. code-block:: python

        handler = FilenameCollisionHandler()
        filename1 = handler.get_unique_filename("Person.parquet", Path("./out"))
        # Returns: "Person.parquet"
        filename2 = handler.get_unique_filename("Person.parquet", Path("./out"))
        # Returns: "Person_1.parquet"
        filename3 = handler.get_unique_filename("Person.parquet", Path("./out"))
        # Returns: "Person_2.parquet"
    """

    # Class-level dictionary to track filename collisions across all instances
    _filename_counts: dict[str, int] = {}

    def get_unique_filename(
        self,
        base_filename: str,
        output_path: Union[str, Path],
    ) -> str:
        """Return a unique filename by adding a _n suffix if a collision is detected.

        Args:
            base_filename: The original filename (e.g. "Person.parquet").
            output_path: The output directory path; collisions are tracked per path.

        Returns:
            A unique filename (e.g. "Person.parquet" or "Person_1.parquet").
        """
        path_str = str(Path(output_path).resolve())
        key = f"{path_str}{base_filename}"

        if key not in self._filename_counts:
            self._filename_counts[key] = 0
            return base_filename

        self._filename_counts[key] += 1
        count = self._filename_counts[key]
        if base_filename.endswith(".parquet"):
            name_without_ext = base_filename[: -len(".parquet")]
            return f"{name_without_ext}_{count}.parquet"
        return f"{base_filename}_{count}"

    @classmethod
    def reset(cls) -> None:
        """Clear the collision-tracking state.

        Intended for tests so each run starts with a clean state.
        """
        cls._filename_counts.clear()
