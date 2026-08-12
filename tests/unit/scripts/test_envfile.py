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
"""Unit tests for reading the examples' .env file.

Every test passes an explicit path. The module's defaults point at the
developer's real .env, and a test that forgot would read it.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
from examples_setup.envfile import apply_env_file, env_value, read_env_file


def test_read_env_file_ignores_comments_and_blanks() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / ".env"
        path.write_text("# a comment\n\nNEO4J_USER=neo4j\nnot-a-pair\n")
        assert read_env_file(path) == {"NEO4J_USER": "neo4j"}


def test_read_env_file_on_a_missing_file_is_empty() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        assert read_env_file(Path(tmpdir) / "nope") == {}


def test_read_env_file_strips_surrounding_quotes() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / ".env"
        path.write_text("A='single'\nB=\"double\"\n")
        assert read_env_file(path) == {"A": "single", "B": "double"}


def test_env_value_prefers_the_process_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A value exported in the shell wins over one sitting in .env."""
    monkeypatch.setenv("OPENAI_API_KEY", "from-shell")
    assert (
        env_value("OPENAI_API_KEY", {"OPENAI_API_KEY": "from-dotenv"}) == "from-shell"
    )


def test_env_value_falls_back_to_the_env_file(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert (
        env_value("OPENAI_API_KEY", {"OPENAI_API_KEY": "from-dotenv"}) == "from-dotenv"
    )


def test_env_value_is_none_when_set_nowhere(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NOT_SET_ANYWHERE", raising=False)
    assert env_value("NOT_SET_ANYWHERE", {}) is None


def test_apply_env_file_does_not_override_the_shell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Probes that go through a third-party SDK read the real environment.

    Overlaying .env lets them see it, but a value the user exported deliberately
    must still win.
    """
    monkeypatch.setenv("ALREADY_SET", "from-shell")
    monkeypatch.delenv("ONLY_IN_DOTENV", raising=False)

    apply_env_file({"ALREADY_SET": "from-dotenv", "ONLY_IN_DOTENV": "value"})

    assert os.environ["ALREADY_SET"] == "from-shell"
    assert os.environ["ONLY_IN_DOTENV"] == "value"
