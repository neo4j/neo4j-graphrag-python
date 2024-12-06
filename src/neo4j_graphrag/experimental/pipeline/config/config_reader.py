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
"""Read JSON or YAML files and returns a dict.
No data validation performed at this stage.
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional

import fsspec
import yaml
from fsspec.implementations.local import LocalFileSystem

logger = logging.getLogger(__name__)


class ConfigReader:
    """Reads config from a file (JSON or YAML format)
    and returns a dict.

    File format is guessed from the extension. Supported extensions are
    (lower or upper case):

    - .json
    - .yaml, .yml

    Example:

    .. code-block:: python

        from pathlib import Path
        from neo4j_graphrag.experimental.pipeline.config.reader import ConfigReader
        reader = ConfigReader()
        reader.read(Path("my_file.json"))

    If reading a file with a different extension but still in JSON or YAML format,
    it is possible to call directly the `read_json` or `read_yaml` methods:

    .. code-block:: python

        reader.read_yaml(Path("my_file.txt"))

    """

    def __init__(self, fs: Optional[fsspec.AbstractFileSystem] = None) -> None:
        self.fs = fs or LocalFileSystem()

    def read_json(self, file_path: str) -> Any:
        logger.debug(f"CONFIG_READER: read from json {file_path}")
        with self.fs.open(file_path, "r") as f:
            return json.load(f)

    def read_yaml(self, file_path: str) -> Any:
        logger.debug(f"CONFIG_READER: read from yaml {file_path}")
        with self.fs.open(file_path, "r") as f:
            return yaml.safe_load(f)

    def _guess_format_and_read(self, file_path: str) -> dict[str, Any]:
        p = Path(file_path)
        extension = p.suffix.lower()
        # Note: .suffix returns an empty string if Path has no extension
        # if not returning a dict, parsing will fail later on
        if extension in [".json"]:
            return self.read_json(file_path)  # type: ignore[no-any-return]
        if extension in [".yaml", ".yml"]:
            return self.read_yaml(file_path)  # type: ignore[no-any-return]
        raise ValueError(f"Unsupported extension: {extension}")

    def read(self, file_path: str) -> dict[str, Any]:
        data = self._guess_format_and_read(file_path)
        return data
