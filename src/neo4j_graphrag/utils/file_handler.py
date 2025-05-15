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
import json
import logging
from pathlib import Path
from typing import Any, Optional, Union

import fsspec
import yaml
from fsspec.implementations.local import LocalFileSystem

logger = logging.getLogger(__name__)


class FileHandler:
    """Utility class to read JSON or YAML files.

    File format is guessed from the extension. Supported extensions are
    (lower or upper case):

    - .json
    - .yaml, .yml

    Example:

    .. code-block:: python

        from pathlib import Path
        from neo4j_graphrag.utils.file_handler import FileHandler
        handler = FileHandler()
        handler.read(Path("my_file.json"))

    If reading a file with a different extension but still in JSON or YAML format,
    it is possible to call directly the `read_json` or `read_yaml` methods:

    .. code-block:: python

        handler.read_yaml(Path("my_file.txt"))

    """

    def __init__(self, fs: Optional[fsspec.AbstractFileSystem] = None) -> None:
        self.fs = fs or LocalFileSystem()

    def read_json(self, file_path: Union[str, Path]) -> Any:
        logger.debug(f"FILE_HANDLER: read from json {file_path}")
        path = self._check_file_exists(file_path)
        with self.fs.open(str(path), "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError("Invalid JSON file") from e

    def read_yaml(self, file_path: Union[str, Path]) -> Any:
        logger.debug(f"FILE_HANDLER: read from yaml {file_path}")
        path = self._check_file_exists(file_path)
        with self.fs.open(str(path), "r") as f:
            try:
                return yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise ValueError("Invalid YAML file") from e

    def _check_file_exists(self, path: Union[str, Path]) -> Path:
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        return file_path

    def _guess_format_and_read(self, file_path: Path) -> Any:
        extension = file_path.suffix.lower()
        # Note: .suffix returns an empty string if Path has no extension
        path_as_string = str(file_path)
        if extension in [".json"]:
            return self.read_json(path_as_string)
        if extension in [".yaml", ".yml"]:
            return self.read_yaml(path_as_string)
        raise ValueError(f"Unsupported extension: {extension}")

    def read(self, file_path: Union[Path, str]) -> Any:
        path = Path(file_path)
        data = self._guess_format_and_read(path)
        return data
