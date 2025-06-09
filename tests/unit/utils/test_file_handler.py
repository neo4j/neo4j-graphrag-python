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
from pathlib import Path
from unittest.mock import patch, Mock, mock_open

import pytest

from neo4j_graphrag.utils.file_handler import FileHandler, FileFormat


def test_file_handler_guess_format() -> None:
    handler = FileHandler()

    assert handler._guess_file_format(Path("file.json")) == FileFormat.JSON
    assert handler._guess_file_format(Path("file.JSON")) == FileFormat.JSON
    assert handler._guess_file_format(Path("file.yaml")) == FileFormat.YAML
    assert handler._guess_file_format(Path("file.YAML")) == FileFormat.YAML
    assert handler._guess_file_format(Path("file.yml")) == FileFormat.YAML
    assert handler._guess_file_format(Path("file.YML")) == FileFormat.YAML
    assert handler._guess_file_format(Path("file.txt")) is None


@patch("neo4j_graphrag.utils.file_handler.FileHandler._read_json")
@patch("neo4j_graphrag.utils.file_handler.FileHandler._check_file_exists")
def test_file_handler_read_json_from_read_method_happy_path(
    mock_file_exists: Mock,
    mock_read_json: Mock,
) -> None:
    handler = FileHandler()

    mock_file_exists.return_value = Path("file.json")
    mock_read_json.return_value = {}
    data = handler.read("file.json")
    mock_read_json.assert_called_with(Path("file.json"))
    assert data == {}

    mock_file_exists.return_value = Path("file.JSON")
    mock_read_json.return_value = {}
    data = handler.read("file.JSON")
    mock_read_json.assert_called_with(Path("file.JSON"))
    assert data == {}


@patch("neo4j_graphrag.utils.file_handler.FileHandler._read_yaml")
@patch("neo4j_graphrag.utils.file_handler.FileHandler._check_file_exists")
def test_file_handler_read_yaml_from_read_method_happy_path(
    mock_file_exists: Mock,
    mock_read_yaml: Mock,
) -> None:
    mock_file_exists.return_value = Path("file.yaml")
    handler = FileHandler()
    mock_read_yaml.return_value = {}
    data = handler.read("file.yaml")
    mock_read_yaml.assert_called_with(Path("file.yaml"))
    assert data == {}

    mock_file_exists.return_value = Path("file.yml")
    mock_read_yaml.return_value = {}
    data = handler.read("file.yml")
    mock_read_yaml.assert_called_with(Path("file.yml"))
    assert data == {}

    mock_file_exists.return_value = Path("file.YAML")
    mock_read_yaml.return_value = {}
    data = handler.read("file.YAML")
    mock_read_yaml.assert_called_with(Path("file.YAML"))
    assert data == {}


@patch("neo4j_graphrag.utils.file_handler.LocalFileSystem")
def test_file_handler_read_json_method_happy_path(
    mock_fs: Mock,
) -> None:
    mock_fs_open = mock_open(read_data='{"data": 1}')
    mock_fs.return_value.open = mock_fs_open

    handler = FileHandler()
    data = handler._read_json(Path("file.json"))
    mock_fs_open.assert_called_once_with("file.json", "r")
    assert data == {"data": 1}


@patch("neo4j_graphrag.utils.file_handler.LocalFileSystem")
def test_file_handler_read_yaml_method_happy_path(
    mock_fs: Mock,
) -> None:
    mock_fs_open = mock_open(
        read_data="""
    data: 1
    """
    )
    mock_fs.return_value.open = mock_fs_open

    handler = FileHandler()
    data = handler._read_yaml(Path("file.yaml"))
    mock_fs_open.assert_called_once_with("file.yaml", "r")
    assert data == {"data": 1}


@patch("neo4j_graphrag.utils.file_handler.LocalFileSystem")
def test_file_handler_read_json_file_does_not_exist(
    mock_fs: Mock,
) -> None:
    mock_fs.return_value.open.side_effect = FileNotFoundError
    handler = FileHandler()
    with pytest.raises(FileNotFoundError):
        handler._read_json(Path("file.json"))


@patch("neo4j_graphrag.utils.file_handler.LocalFileSystem")
def test_file_handler_read_json_invalid_json(
    mock_fs: Mock,
) -> None:
    mock_fs_open = mock_open(read_data="{")
    mock_fs.return_value.open = mock_fs_open

    handler = FileHandler()
    with pytest.raises(ValueError, match="Invalid JSON"):
        handler._read_json(Path("file.json"))


@patch("neo4j_graphrag.utils.file_handler.LocalFileSystem")
def test_file_handler_read_yaml_file_does_not_exist(
    mock_fs: Mock,
) -> None:
    mock_fs.return_value.open.side_effect = FileNotFoundError
    handler = FileHandler()
    with pytest.raises(FileNotFoundError):
        handler._read_yaml(Path("file.yaml"))


@patch("neo4j_graphrag.utils.file_handler.LocalFileSystem")
def test_file_handler_read_yaml_invalid_yaml(
    mock_fs: Mock,
) -> None:
    mock_fs_open = mock_open(
        read_data="""
    data: [
    """
    )
    mock_fs.return_value.open = mock_fs_open

    handler = FileHandler()
    with pytest.raises(ValueError, match="Invalid YAML"):
        handler._read_yaml(Path("file.yaml"))


@patch("neo4j_graphrag.utils.file_handler.FileHandler._check_file_exists")
def test_file_handler_file_can_be_written_file(mock_file_exists: Mock) -> None:
    # file does not exist
    mock_file_exists.side_effect = FileNotFoundError()
    handler = FileHandler()
    # nothing happens = all good
    handler._check_file_can_be_written(Path("file.json"))
    # file exist, overwrite is False
    mock_file_exists.side_effect = None
    handler = FileHandler()
    with pytest.raises(ValueError):
        handler._check_file_can_be_written(Path("file.json"), overwrite=False)
    # file exists, overwrite is True
    mock_file_exists.side_effect = None
    handler = FileHandler()
    # nothing happens = all good
    handler._check_file_can_be_written(Path("file.json"), overwrite=True)


@patch("neo4j_graphrag.utils.file_handler.LocalFileSystem")
@patch("neo4j_graphrag.utils.file_handler.json")
def test_file_handler_write_json_happy_path(
    mock_json_module: Mock,
    mock_fs: Mock,
) -> None:
    mock_fs_open = mock_open()
    mock_fs.return_value.open = mock_fs_open
    file_handler = FileHandler()
    file_handler._write_json({"some": "data"}, Path("file.json"))
    mock_fs_open.assert_called_once_with("file.json", "w")
    mock_json_module.dump.assert_called_with(
        {"some": "data"}, mock_fs_open.return_value, indent=2
    )


@patch("neo4j_graphrag.utils.file_handler.LocalFileSystem")
@patch("neo4j_graphrag.utils.file_handler.json")
def test_file_handler_write_json_extra_kwargs_happy_path(
    mock_json_module: Mock,
    mock_fs: Mock,
) -> None:
    mock_fs_open = mock_open()
    mock_fs.return_value.open = mock_fs_open
    file_handler = FileHandler()
    file_handler._write_json({"some": "data"}, Path("file.json"), indent=4, default=str)
    mock_fs_open.assert_called_once_with("file.json", "w")
    mock_json_module.dump.assert_called_with(
        {"some": "data"}, mock_fs_open.return_value, indent=4, default=str
    )


@patch("neo4j_graphrag.utils.file_handler.LocalFileSystem")
@patch("neo4j_graphrag.utils.file_handler.yaml")
def test_file_handler_write_yaml_happy_path(
    mock_yaml_module: Mock,
    mock_fs: Mock,
) -> None:
    mock_fs_open = mock_open()
    mock_fs.return_value.open = mock_fs_open
    file_handler = FileHandler()
    file_handler._write_yaml({"some": "data"}, Path("file.yaml"))
    mock_fs_open.assert_called_once_with("file.yaml", "w")
    mock_yaml_module.safe_dump.assert_called_with(
        {"some": "data"},
        mock_fs_open.return_value,
        default_flow_style=False,
        sort_keys=True,
    )


@patch("neo4j_graphrag.utils.file_handler.LocalFileSystem")
@patch("neo4j_graphrag.utils.file_handler.yaml")
def test_file_handler_write_yaml_extra_kwargs_happy_path(
    mock_yaml_module: Mock,
    mock_fs: Mock,
) -> None:
    mock_fs_open = mock_open()
    mock_fs.return_value.open = mock_fs_open
    file_handler = FileHandler()
    file_handler._write_yaml(
        {"some": "data"}, Path("file.json"), default_flow_style="toto", other_keyword=42
    )
    mock_fs_open.assert_called_once_with("file.json", "w")
    mock_yaml_module.safe_dump.assert_called_with(
        {"some": "data"},
        mock_fs_open.return_value,
        default_flow_style="toto",
        sort_keys=True,
        other_keyword=42,
    )


@patch("neo4j_graphrag.utils.file_handler.FileHandler._write_json")
@patch("neo4j_graphrag.utils.file_handler.FileHandler._check_file_can_be_written")
def test_file_handler_write_json_from_write_method_happy_path(
    mock_file_can_be_written: Mock,
    mock_write_json: Mock,
) -> None:
    handler = FileHandler()
    handler.write("data", "file.json")
    mock_write_json.assert_called_with("data", Path("file.json"))


@patch("neo4j_graphrag.utils.file_handler.FileHandler._write_yaml")
@patch("neo4j_graphrag.utils.file_handler.FileHandler._check_file_can_be_written")
def test_file_handler_write_yaml_from_write_method_happy_path(
    mock_file_can_be_written: Mock,
    mock_write_yaml: Mock,
) -> None:
    handler = FileHandler()
    handler.write("data", "file.yaml")
    mock_write_yaml.assert_called_with("data", Path("file.yaml"))


@patch("neo4j_graphrag.utils.file_handler.FileHandler._write_yaml")
@patch("neo4j_graphrag.utils.file_handler.FileHandler._check_file_can_be_written")
def test_file_handler_write_yaml_from_write_method_overwrite_format_happy_path(
    mock_file_can_be_written: Mock,
    mock_write_yaml: Mock,
) -> None:
    handler = FileHandler()
    handler.write("data", "file.txt", format=FileFormat.YAML)
    mock_write_yaml.assert_called_with("data", Path("file.txt"))


@patch("neo4j_graphrag.utils.file_handler.FileHandler._write_json")
@patch("neo4j_graphrag.utils.file_handler.FileHandler._check_file_can_be_written")
def test_file_handler_write_json_from_write_method_overwrite_format_happy_path(
    mock_file_can_be_written: Mock,
    mock_write_json: Mock,
) -> None:
    handler = FileHandler()
    handler.write("data", "file.txt", format=FileFormat.JSON)
    mock_write_json.assert_called_with("data", Path("file.txt"))
