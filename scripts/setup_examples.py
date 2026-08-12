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
"""Walk through getting an environment together for running the examples.

    python scripts/setup_examples.py                  # walk through everything
    python scripts/setup_examples.py --provider gemini # just one provider's key
    python scripts/check_setup.py                     # what is still missing

Works in tiers, so the free and local providers work before any cloud account is
involved: Python extras and a local Neo4j first, then API keys, then the cloud
logins. Every step is skippable and safe to re-run.

Keys are read with no terminal echo, validated against the provider with a free
list-models call, and written only to a gitignored ``.env`` created at mode 0600.
A key that does not validate is never written - a bad value there turns a clear
"unset" into a confusing 401 later. Nothing is ever printed in full.

``scripts/check_setup.py`` reports what is missing without changing anything;
run it first if you only want to know where you stand.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from examples_setup import ENV_FILE  # noqa: E402
from examples_setup.cloud import cloud_logins  # noqa: E402
from examples_setup.console import BOLD, DIM, colour  # noqa: E402
from examples_setup.installer import tier_0_base, tier_1_keys  # noqa: E402
from examples_setup.providers import PROVIDERS  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--provider",
        help="only configure this provider's API key, e.g. gemini",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="never prompt; take the default for every question",
    )
    args = parser.parse_args()

    if args.provider and args.provider not in PROVIDERS:
        print(f"unknown provider '{args.provider}'", file=sys.stderr)
        print(f"choose from: {', '.join(sorted(PROVIDERS))}", file=sys.stderr)
        return 2

    assume_yes = bool(args.non_interactive)
    print(colour("neo4j-graphrag examples setup", BOLD))
    print(
        colour(
            f"Keys are written only to {ENV_FILE.name} (mode 0600, gitignored) "
            "and never printed in full.",
            DIM,
        )
    )

    if args.provider:
        # Narrowing to one provider only makes sense for the key step.
        tier_1_keys(assume_yes, args.provider)
    else:
        tier_0_base(assume_yes)
        tier_1_keys(assume_yes, None)
        cloud_logins(assume_yes)

    print()
    print(colour("Done. Check the result with:", BOLD))
    print("  python scripts/check_setup.py")
    print()
    print(
        colour(
            "Most examples read the environment directly, so export .env "
            "before running one:\n  set -a; source .env; set +a",
            DIM,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
