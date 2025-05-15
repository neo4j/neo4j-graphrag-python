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

    JSON_VALID_EXTENSIONS = (".json",)
    YALM_VALID_EXTENSIONS = (".yaml", ".yml")

    def __init__(self, fs: Optional[fsspec.AbstractFileSystem] = None) -> None:
        self.fs = fs or LocalFileSystem()

    def _get_file_extension(self, path: Union[str, Path]) -> str:
        p = Path(path)
        extension = p.suffix.lower()
        return extension

    def _check_file_exists(self, path: Union[str, Path]) -> Path:
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        return file_path

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

    def _guess_format_and_read(self, file_path: Union[str, Path]) -> Any:
        extension = self._get_file_extension(file_path)
        # Note: .suffix returns an empty string if Path has no extension
        if extension in self.JSON_VALID_EXTENSIONS:
            return self.read_json(file_path)
        if extension in self.YALM_VALID_EXTENSIONS:
            return self.read_yaml(file_path)
        raise ValueError(f"Unsupported extension: {extension}")

    def read(self, file_path: Union[Path, str]) -> Any:
        data = self._guess_format_and_read(file_path)
        return data

    def _check_file_can_be_written(
        self, path: Union[str, Path], overwrite: bool = False
    ) -> None:
        if overwrite:
            # we can overwrite, so no matter if file already exists or not
            return
        try:
            self._check_file_exists(path)
            # did not raise, meaning the file exists
            raise ValueError("File already exists. Use overwrite=True to overwrite.")
        except FileNotFoundError:
            # file not found all godo
            pass

    def write_json(
        self,
        data: Any,
        file_path: Union[Path, str],
        overwrite: bool = False,
        **extra_kwargs: Any,
    ) -> None:
        self._check_file_can_be_written(file_path, overwrite)
        fp = str(file_path)
        kwargs: dict[str, Any] = {
            "indent": 2,
        }
        kwargs.update(extra_kwargs)
        with self.fs.open(fp, "w") as f:
            json.dump(data, f, **kwargs)

    def write_yaml(
        self,
        data: Any,
        file_path: Union[Path, str],
        overwrite: bool = False,
        **extra_kwargs: Any,
    ) -> None:
        self._check_file_can_be_written(file_path, overwrite)
        fp = str(file_path)
        kwargs: dict[str, Any] = {
            "default_flow_style": False,
            "sort_keys": True,
        }
        kwargs.update(extra_kwargs)
        with self.fs.open(fp, "w") as f:
            yaml.safe_dump(data, f, **kwargs)

    def write(
        self,
        data: Any,
        file_path: Union[Path, str],
        overwrite: bool = False,
        **extra_kwargs: Any,
    ) -> None:
        extension = self._get_file_extension(file_path)
        if extension in self.JSON_VALID_EXTENSIONS:
            return self.write_json(data, file_path, overwrite=overwrite, **extra_kwargs)
        if extension in self.YALM_VALID_EXTENSIONS:
            return self.write_yaml(data, file_path, overwrite=overwrite, **extra_kwargs)
        raise ValueError(f"Unsupported extension: {extension}")
