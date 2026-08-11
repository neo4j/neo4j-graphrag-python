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
"""Reading and writing the .env file.

Credentials live here and nowhere else: the file is gitignored, written at
mode 0600, and its contents are never echoed back in full.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from . import ENV_FILE


def read_env_file(path: Path = ENV_FILE) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, _, value = stripped.partition("=")
        values[name.strip()] = value.strip().strip("'\"")
    return values


def env_value(name: str, env_file: dict[str, str]) -> Optional[str]:
    """Resolve an env var from the process first, then .env."""
    return os.environ.get(name) or env_file.get(name) or None


def apply_env_file(env_file: dict[str, str]) -> None:
    """Overlay .env onto the process, without overriding what is already set.

    Needed because probes that go through a third-party SDK - botocore resolving
    AWS_PROFILE, for instance - read the real environment and cannot see a value
    that only exists in .env. Without this the doctor would report "no
    credentials" for a profile the user had correctly configured.
    """
    for name, value in env_file.items():
        if value and name not in os.environ:
            os.environ[name] = value


def mask(value: str) -> str:
    """Show just enough to recognise a key, never enough to use it."""
    if len(value) <= 8:
        return "*" * len(value)
    return f"{'*' * 8}{value[-4:]}"


def write_env_var(name: str, value: str, path: Path = ENV_FILE) -> None:
    """Set one variable in .env, preserving everything else. Mode 0600.

    Also matches the commented-out placeholder the template ships, so setting a
    key replaces its placeholder in place rather than appending a duplicate.
    """
    lines = path.read_text().splitlines() if path.exists() else []
    replaced = False
    for index, line in enumerate(lines):
        stripped = line.strip()
        # One leading '#' is the template's own placeholder. More than one is a
        # deliberate comment, so leave it alone rather than overwrite it.
        commented = stripped[1:].strip() if stripped.startswith("#") else stripped
        if stripped.startswith(f"{name}=") or commented.startswith(f"{name}="):
            lines[index] = f"{name}={value}"
            replaced = True
            break
    if not replaced:
        if lines and lines[-1].strip():
            lines.append("")
        lines.append(f"{name}={value}")

    _write_private(path, "\n".join(lines) + "\n")


def _write_private(path: Path, content: str) -> None:
    """Replace path's contents, never leaving a secret readable by anyone else.

    Written through a temp file in the same directory so an interrupted write
    cannot truncate the credentials already in place, and created at 0600 up
    front - creating first and chmod-ing after leaves the secret world-readable
    for the window in between.
    """
    tmp = path.with_name(f".{path.name}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(content)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
