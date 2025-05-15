from pathlib import Path
from unittest.mock import patch, Mock, mock_open

import pytest

from neo4j_graphrag.utils.file_handler import FileHandler


@patch("neo4j_graphrag.utils.file_handler.FileHandler.read_json")
def test_file_handler_read_json_from_read_method_happy_path(
    mock_read_json: Mock,
) -> None:
    handler = FileHandler()

    mock_read_json.return_value = {}
    data = handler.read("file.json")
    mock_read_json.assert_called_with("file.json")
    assert data == {}

    mock_read_json.return_value = {}
    data = handler.read("file.JSON")
    mock_read_json.assert_called_with("file.JSON")
    assert data == {}


@patch("neo4j_graphrag.utils.file_handler.FileHandler.read_yaml")
def test_file_handler_read_yaml_from_read_method_happy_path(
    mock_read_yaml: Mock,
) -> None:
    handler = FileHandler()
    mock_read_yaml.return_value = {}
    data = handler.read("file.yaml")
    mock_read_yaml.assert_called_with("file.yaml")
    assert data == {}

    mock_read_yaml.return_value = {}
    data = handler.read("file.yml")
    mock_read_yaml.assert_called_with("file.yml")
    assert data == {}

    mock_read_yaml.return_value = {}
    data = handler.read("file.YAML")
    mock_read_yaml.assert_called_with("file.YAML")
    assert data == {}


@patch("neo4j_graphrag.utils.file_handler.FileHandler._check_file_exists")
@patch("neo4j_graphrag.utils.file_handler.LocalFileSystem")
def test_file_handler_read_json_method_happy_path(
    mock_fs: Mock, mock_file_exists: Mock
) -> None:
    mock_file_exists.return_value = Path("file.json")
    mock_fs_open = mock_open(read_data="{}")
    mock_fs.return_value.open = mock_fs_open

    handler = FileHandler()
    data = handler.read_json("file.json")
    mock_fs_open.assert_called_once_with("file.json", "r")
    assert data == {}


@patch("neo4j_graphrag.utils.file_handler.FileHandler._check_file_exists")
@patch("neo4j_graphrag.utils.file_handler.LocalFileSystem")
def test_file_handler_read_yaml_method_happy_path(
    mock_fs: Mock, mock_file_exists: Mock
) -> None:
    mock_file_exists.return_value = Path("file.yaml")
    mock_fs_open = mock_open(
        read_data="""
    data: 1
    """
    )
    mock_fs.return_value.open = mock_fs_open

    handler = FileHandler()
    data = handler.read_yaml("file.yaml")
    mock_fs_open.assert_called_once_with("file.yaml", "r")
    assert data == {"data": 1}


@patch("neo4j_graphrag.utils.file_handler.FileHandler._check_file_exists")
def test_file_handler_read_json_file_does_not_exist(mock_file_exists: Mock) -> None:
    mock_file_exists.side_effect = FileNotFoundError()

    handler = FileHandler()
    with pytest.raises(FileNotFoundError):
        handler.read_json("file.json")


@patch("neo4j_graphrag.utils.file_handler.FileHandler._check_file_exists")
@patch("neo4j_graphrag.utils.file_handler.LocalFileSystem")
def test_file_handler_read_json_invalid_json(
    mock_fs: Mock, mock_file_exists: Mock
) -> None:
    mock_file_exists.return_value = True
    mock_fs_open = mock_open(read_data="{")
    mock_fs.return_value.open = mock_fs_open

    handler = FileHandler()
    with pytest.raises(ValueError, match="Invalid JSON"):
        handler.read_json("file.json")


@patch("neo4j_graphrag.utils.file_handler.FileHandler._check_file_exists")
def test_file_handler_read_yaml_file_does_not_exist(mock_file_exists: Mock) -> None:
    mock_file_exists.side_effect = FileNotFoundError()

    handler = FileHandler()
    with pytest.raises(FileNotFoundError):
        handler.read_yaml("file.yaml")


@patch("neo4j_graphrag.utils.file_handler.FileHandler._check_file_exists")
@patch("neo4j_graphrag.utils.file_handler.LocalFileSystem")
def test_file_handler_read_yaml_invalid_yaml(
    mock_fs: Mock, mock_file_exists: Mock
) -> None:
    mock_file_exists.return_value = True
    mock_fs_open = mock_open(
        read_data="""
    data: [
    """
    )
    mock_fs.return_value.open = mock_fs_open

    handler = FileHandler()
    with pytest.raises(ValueError, match="Invalid YAML"):
        handler.read_yaml("file.yaml")


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
@patch("neo4j_graphrag.utils.file_handler.FileHandler._check_file_can_be_written")
@patch("neo4j_graphrag.utils.file_handler.json")
def test_file_handler_write_json_happy_path(
    mock_json_module: Mock,
    mock_file_can_be_written: Mock,
    mock_fs: Mock,
) -> None:
    mock_fs_open = mock_open()
    mock_fs.return_value.open = mock_fs_open
    file_handler = FileHandler()
    file_handler.write_json({"some": "data"}, "file.json")
    mock_fs_open.assert_called_once_with("file.json", "w")
    mock_json_module.dump.assert_called_with(
        {"some": "data"}, mock_fs_open.return_value, indent=2
    )


@patch("neo4j_graphrag.utils.file_handler.LocalFileSystem")
@patch("neo4j_graphrag.utils.file_handler.FileHandler._check_file_can_be_written")
@patch("neo4j_graphrag.utils.file_handler.json")
def test_file_handler_write_json_extra_kwargs_happy_path(
    mock_json_module: Mock,
    mock_file_can_be_written: Mock,
    mock_fs: Mock,
) -> None:
    mock_fs_open = mock_open()
    mock_fs.return_value.open = mock_fs_open
    file_handler = FileHandler()
    file_handler.write_json({"some": "data"}, "file.json", indent=4, default=str)
    mock_fs_open.assert_called_once_with("file.json", "w")
    mock_json_module.dump.assert_called_with(
        {"some": "data"}, mock_fs_open.return_value, indent=4, default=str
    )


@patch("neo4j_graphrag.utils.file_handler.LocalFileSystem")
@patch("neo4j_graphrag.utils.file_handler.FileHandler._check_file_can_be_written")
@patch("neo4j_graphrag.utils.file_handler.yaml")
def test_file_handler_write_yaml_happy_path(
    mock_yaml_module: Mock,
    mock_file_can_be_written: Mock,
    mock_fs: Mock,
) -> None:
    mock_fs_open = mock_open()
    mock_fs.return_value.open = mock_fs_open
    file_handler = FileHandler()
    file_handler.write_yaml({"some": "data"}, "file.yaml")
    mock_fs_open.assert_called_once_with("file.yaml", "w")
    mock_yaml_module.safe_dump.assert_called_with(
        {"some": "data"},
        mock_fs_open.return_value,
        default_flow_style=False,
        sort_keys=True,
    )


@patch("neo4j_graphrag.utils.file_handler.LocalFileSystem")
@patch("neo4j_graphrag.utils.file_handler.FileHandler._check_file_can_be_written")
@patch("neo4j_graphrag.utils.file_handler.yaml")
def test_file_handler_write_yaml_extra_kwargs_happy_path(
    mock_yaml_module: Mock,
    mock_file_can_be_written: Mock,
    mock_fs: Mock,
) -> None:
    mock_fs_open = mock_open()
    mock_fs.return_value.open = mock_fs_open
    file_handler = FileHandler()
    file_handler.write_yaml(
        {"some": "data"}, "file.json", default_flow_style="toto", other_keyword=42
    )
    mock_fs_open.assert_called_once_with("file.json", "w")
    mock_yaml_module.safe_dump.assert_called_with(
        {"some": "data"},
        mock_fs_open.return_value,
        default_flow_style="toto",
        sort_keys=True,
        other_keyword=42,
    )


@patch("neo4j_graphrag.utils.file_handler.FileHandler.write_json")
def test_file_handler_write_json_from_write_method_happy_path(
    mock_write_json: Mock,
) -> None:
    handler = FileHandler()
    handler.write("data", "file.json")
    mock_write_json.assert_called_with("data", "file.json", overwrite=False)


@patch("neo4j_graphrag.utils.file_handler.FileHandler.write_yaml")
def test_file_handler_write_yaml_from_write_method_happy_path(
    mock_write_yaml: Mock,
) -> None:
    handler = FileHandler()
    handler.write("data", "file.yaml")
    mock_write_yaml.assert_called_with("data", "file.yaml", overwrite=False)
