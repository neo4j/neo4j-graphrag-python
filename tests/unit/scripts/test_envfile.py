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
"""Unit tests for the .env writer used by the examples installer.

Every test passes an explicit path. The module's defaults point at the
developer's real .env, and a test that forgot would rewrite it.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from examples_setup.envfile import mask, read_env_file, write_env_var

SECRET = "sk-test-0123456789abcdef"


def test_a_new_env_file_is_never_world_readable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The secret must not exist on disk at default permissions, even briefly.

    Regression: the file was created by write_text() and chmod-ed afterwards, so
    a fresh .env held the key at 0644 until the next statement ran. Asserting the
    final mode cannot see that window - both versions end at 0600 - so this
    forbids the fixup instead: the mode has to be right at creation.
    """

    def no_chmod(*args: object, **kwargs: object) -> None:
        raise AssertionError("mode must be set when the file is created")

    monkeypatch.setattr(Path, "chmod", no_chmod)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / ".env"
        write_env_var("OPENAI_API_KEY", SECRET, path)
        assert path.stat().st_mode & 0o777 == 0o600
        assert path.read_text().strip() == f"OPENAI_API_KEY={SECRET}"


def test_an_existing_env_file_stays_private() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / ".env"
        write_env_var("OPENAI_API_KEY", SECRET, path)
        write_env_var("COHERE_API_KEY", "co-second", path)
        assert path.stat().st_mode & 0o777 == 0o600


def test_no_temp_file_is_left_behind() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        directory = Path(tmpdir)
        write_env_var("OPENAI_API_KEY", SECRET, directory / ".env")
        assert [p.name for p in directory.iterdir()] == [".env"]


def test_other_variables_are_preserved() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / ".env"
        path.write_text("NEO4J_URI=bolt://localhost:7687\nNEO4J_USER=neo4j\n")
        write_env_var("OPENAI_API_KEY", SECRET, path)
        values = read_env_file(path)
        assert values["NEO4J_URI"] == "bolt://localhost:7687"
        assert values["NEO4J_USER"] == "neo4j"
        assert values["OPENAI_API_KEY"] == SECRET


def test_the_commented_placeholder_is_replaced_in_place() -> None:
    """The template ships '# OPENAI_API_KEY=', which must not be duplicated."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / ".env"
        path.write_text("# OPENAI_API_KEY=\nNEO4J_USER=neo4j\n")
        write_env_var("OPENAI_API_KEY", SECRET, path)
        lines = path.read_text().splitlines()
        assert lines == [f"OPENAI_API_KEY={SECRET}", "NEO4J_USER=neo4j"]


def test_a_deliberately_commented_out_line_is_not_overwritten() -> None:
    """One '#' is the template's placeholder; two is somebody's decision."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / ".env"
        path.write_text("## OPENAI_API_KEY=do-not-use\n")
        write_env_var("OPENAI_API_KEY", SECRET, path)
        lines = path.read_text().splitlines()
        assert "## OPENAI_API_KEY=do-not-use" in lines
        assert f"OPENAI_API_KEY={SECRET}" in lines


def test_setting_the_same_variable_twice_does_not_duplicate_it() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / ".env"
        write_env_var("OPENAI_API_KEY", "first", path)
        write_env_var("OPENAI_API_KEY", "second", path)
        contents = path.read_text()
        assert contents.count("OPENAI_API_KEY=") == 1
        assert read_env_file(path)["OPENAI_API_KEY"] == "second"


def test_read_env_file_ignores_comments_and_blanks() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / ".env"
        path.write_text("# a comment\n\nNEO4J_USER=neo4j\nnot-a-pair\n")
        assert read_env_file(path) == {"NEO4J_USER": "neo4j"}


def test_read_env_file_on_a_missing_file_is_empty() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        assert read_env_file(Path(tmpdir) / "nope") == {}


@pytest.mark.parametrize(
    "value",
    [
        "sk-proj-abcdefghijklmnop",
        "short",
        "",
        "12345678",
    ],
)
def test_mask_never_reveals_more_than_the_last_four_characters(value: str) -> None:
    masked = mask(value)
    assert value not in masked or value == ""
    assert masked.replace("*", "") == (value[-4:] if len(value) > 8 else "")
