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
"""Get an environment together for running the examples.

    python scripts/setup_examples.py --check     # what is missing, changes nothing
    python scripts/setup_examples.py             # walk through fixing it

Two modes over one model of what each example needs (see
``example_requirements.py``):

**Doctor** (``--check``) probes packages, env vars, databases and containers and
reports, per example, the single thing standing in the way. It writes nothing,
touches no network beyond a TCP connect, and exits 0 unless ``--strict``.

**Installer** (default) walks the same ground interactively, in tiers, so the
free and local providers work before any cloud account is involved:

    tier 0  Python extras, local Neo4j, the indexes examples assume exist
    tier 1  API keys - Gemini, Cohere and Mistral are free without a card
    tier 2  local runtimes - Ollama, sentence-transformers, spaCy, vector stores
    tier 3  cloud - Vertex AI, Bedrock, Azure OpenAI

Every step is skippable and safe to re-run.

Credentials
-----------
Keys are read from you with no terminal echo, validated, then written only to
``.env`` at the repo root, mode 0600, which is gitignored. They are never echoed
back in full, never logged, and never written to a tracked file. Cloud account
identifiers are resolved at run time and held in memory only. Nothing this
script writes is intended to be committed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from examples_setup import ENV_FILE  # noqa: E402
from examples_setup.cloud import tier_3_cloud  # noqa: E402
from examples_setup.console import BOLD, DIM, colour  # noqa: E402
from examples_setup.doctor import run_doctor  # noqa: E402
from examples_setup.installer import (  # noqa: E402
    tier_0_base,
    tier_1_keys,
    tier_2_local,
)
from examples_setup.providers import PROVIDERS  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="report what is missing and change nothing",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="with --check, exit non-zero if any example is blocked",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="with --check, list every example"
    )
    parser.add_argument(
        "--tier",
        type=int,
        choices=[0, 1, 2, 3],
        action="append",
        help="only run these tiers (repeatable; default: all)",
    )
    parser.add_argument(
        "--provider",
        help="only configure this provider, e.g. gemini",
    )
    parser.add_argument(
        "--cloud-profile",
        choices=["aura", "own"],
        help="tier 3: use a Neo4j Aura dev environment, or your own accounts",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="never prompt; take the default for every question",
    )
    args = parser.parse_args()

    if args.check:
        return run_doctor(strict=args.strict, verbose=args.verbose)

    if args.provider and args.provider not in PROVIDERS:
        print(f"unknown provider '{args.provider}'", file=sys.stderr)
        print(f"choose from: {', '.join(sorted(PROVIDERS))}", file=sys.stderr)
        return 2

    tiers = sorted(set(args.tier)) if args.tier else [0, 1, 2, 3]
    if args.provider and not args.tier:
        tiers = [PROVIDERS[args.provider].tier]

    assume_yes = bool(args.non_interactive)
    print(colour("neo4j-graphrag examples setup", BOLD))
    print(
        colour(
            f"Keys are written only to {ENV_FILE.name} (mode 0600, gitignored) "
            "and never printed in full.",
            DIM,
        )
    )

    if 0 in tiers:
        tier_0_base(assume_yes)
    if 1 in tiers:
        tier_1_keys(assume_yes, args.provider)
    if 2 in tiers:
        tier_2_local(assume_yes, args.provider)
    if 3 in tiers:
        tier_3_cloud(assume_yes, args.cloud_profile)

    print()
    print(colour("Done. Check the result with:", BOLD))
    print("  python scripts/setup_examples.py --check")
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
