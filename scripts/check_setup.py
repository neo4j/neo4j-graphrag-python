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
"""Report what is stopping each example under ``examples/`` from running.

    python scripts/check_setup.py             # what is missing
    python scripts/check_setup.py --verbose    # and list every example
    python scripts/check_setup.py --strict     # exit non-zero if anything is blocked

Resolves every example against the requirement model in
``example_requirements.py``, then probes this machine: which extras are
installed, which env vars are set, which credentials resolve, which services
answer, and what the local Neo4j actually contains.

Reports the single first thing blocking each example, grouped so the biggest
wins come first, with the command that fixes it. Writes nothing, and needs no
credentials of its own. Network use is limited to TCP connects, except that
resolving Google credentials shells out to ``gcloud`` to mint an access token.

``examples/SETUP.md`` covers the same ground as prose, for reading rather than
running.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from examples_setup.doctor import run_doctor  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero if any example is blocked",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="list every example rather than the first few per reason",
    )
    args = parser.parse_args()
    return run_doctor(strict=args.strict, verbose=args.verbose)


if __name__ == "__main__":
    raise SystemExit(main())
