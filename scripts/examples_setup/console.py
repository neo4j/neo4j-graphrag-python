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
"""Terminal output and prompts."""

from __future__ import annotations

import subprocess
import sys
from typing import Sequence

from . import REPO_ROOT

GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


def colour(text: str, code: str) -> str:
    if not sys.stdout.isatty():
        return text
    return f"{code}{text}{RESET}"


def confirm(question: str, default: bool = True, assume_yes: bool = False) -> bool:
    if assume_yes:
        return default
    suffix = "[Y/n]" if default else "[y/N]"
    try:
        answer = input(f"{question} {suffix} ").strip().lower()
    except EOFError:
        return default
    if not answer:
        return default
    return answer.startswith("y")


def run_command(command: Sequence[str], check: bool = False) -> int:
    print(colour(f"  $ {' '.join(command)}", DIM))
    result = subprocess.run(list(command), cwd=REPO_ROOT)
    if check and result.returncode != 0:
        print(colour(f"  command failed ({result.returncode})", RED))
    return result.returncode


def heading(text: str) -> None:
    print()
    print(colour(f"== {text}", BOLD))
