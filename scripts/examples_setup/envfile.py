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
