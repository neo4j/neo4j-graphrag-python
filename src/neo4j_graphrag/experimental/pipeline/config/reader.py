import json
from pathlib import Path
from typing import Any

import yaml


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

    @staticmethod
    def read_json(file_path: Path) -> Any:
        with open(file_path, "r") as f:
            return json.load(f)

    @staticmethod
    def read_yaml(file_path: Path) -> Any:
        with open(file_path, "r") as f:
            return yaml.safe_load(f)

    def _guess_format_and_read(self, file_path: Path) -> dict[str, Any]:
        extension = file_path.suffix.lower()
        # Note: .suffix returns an empty string if Path has no extension
        if extension in [".json"]:
            return self.read_json(file_path)
        if extension in [".yaml", ".yml"]:
            return self.read_yaml(file_path)
        raise ValueError(f"Unsupported extension: {extension}")

    def read(self, file_path: Path) -> dict[str, Any]:
        data = self._guess_format_and_read(file_path)
        return data
