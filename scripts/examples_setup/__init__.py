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
"""Setup tooling for running the examples.

The command-line entry point is ``scripts/setup_examples.py``; this package
holds the pieces it drives. Split by concern so each part can be read on its
own: providers and their credentials, the .env file, environment probes, the
doctor report, and the interactive installer.
"""

from __future__ import annotations

import sys
from pathlib import Path

# example_requirements sits beside this package rather than inside it, because
# check_examples.py imports it too. Importing any submodule runs this first, so
# the path is in place before anything reaches for it.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import example_requirements as reqs  # noqa: E402

REPO_ROOT = reqs.REPO_ROOT
ENV_FILE = REPO_ROOT / ".env"
ENV_TEMPLATE = REPO_ROOT / "examples" / ".env.example"
COMPOSE_FILE = REPO_ROOT / "examples" / "docker-compose.yml"
