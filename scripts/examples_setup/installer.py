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
"""Walk through installing what is missing, in tiers."""

from __future__ import annotations

import subprocess
import sys
from getpass import getpass
from typing import Optional

import example_requirements as reqs

from . import COMPOSE_FILE, ENV_FILE, ENV_TEMPLATE
from .console import BOLD, GREEN, RED, YELLOW, colour, confirm, heading, run_command
from .envfile import env_value, mask, read_env_file, write_env_var
from .probes import command_exists, probe_neo4j
from .providers import (
    LOCAL_FULLTEXT_INDEXES,
    LOCAL_INDEXES,
    PROVIDERS,
    TIER_NAMES,
    Provider,
)


def tier_0_base(assume_yes: bool) -> None:
    heading(f"Tier 0: {TIER_NAMES[0]}")

    if confirm(
        "Install all Python extras with uv sync --all-extras?", True, assume_yes
    ):
        if command_exists("uv"):
            run_command(["uv", "sync", "--all-extras", "--group", "dev"])
        else:
            print(colour("  uv not found - see https://docs.astral.sh/uv/", RED))

    if not ENV_FILE.exists() and ENV_TEMPLATE.exists():
        if confirm("Create .env from examples/.env.example?", True, assume_yes):
            ENV_FILE.write_text(ENV_TEMPLATE.read_text())
            ENV_FILE.chmod(0o600)
            print(f"  wrote {ENV_FILE} (mode 0600, gitignored)")

    if not reqs.service_available(reqs.NEO4J_LOCAL):
        if confirm("Start local Neo4j with Docker Compose?", True, assume_yes):
            if not command_exists("docker"):
                print(colour("  docker not found", RED))
            else:
                run_command(
                    ["docker", "compose", "-f", str(COMPOSE_FILE), "up", "-d", "neo4j"]
                )
                print("  waiting for Neo4j to accept connections...")
                _wait_for(reqs.NEO4J_LOCAL, attempts=30)
    else:
        print("  local Neo4j already reachable")

    env_file = read_env_file()
    state = probe_neo4j(env_file)
    if state.authenticated:
        missing_vector = [i for i in LOCAL_INDEXES if i[0] not in state.indexes]
        missing_fulltext = [
            i for i in LOCAL_FULLTEXT_INDEXES if i[0] not in state.indexes
        ]
        if missing_vector or missing_fulltext:
            names = [i[0] for i in missing_vector] + [i[0] for i in missing_fulltext]
            if confirm(f"Create missing indexes {', '.join(names)}?", True, assume_yes):
                _create_indexes(env_file, missing_vector, missing_fulltext)
        else:
            print("  all expected indexes already exist")
    elif state.reachable:
        print(colour(f"  could not authenticate: {state.error}", YELLOW))


def _wait_for(service: str, attempts: int = 30, delay: float = 2.0) -> bool:
    import time

    host, port = reqs.SERVICE_ENDPOINTS[service]
    for _ in range(attempts):
        if reqs.port_open(host, port):
            print(colour(f"  {service} is up", GREEN))
            return True
        time.sleep(delay)
    print(colour(f"  {service} did not come up in time", RED))
    return False


def _create_indexes(
    env_file: dict[str, str],
    vector: list[tuple[str, str, str, int]],
    fulltext: list[tuple[str, str, str]],
) -> None:
    """Create the indexes examples assume exist.

    Uses the library's own helpers so the index definitions match what the
    retrievers expect, rather than hand-written Cypher that could drift.
    """
    try:
        import neo4j
        from neo4j_graphrag.indexes import create_fulltext_index, create_vector_index
    except ImportError as exc:
        print(colour(f"  cannot create indexes yet ({exc}); run uv sync first", YELLOW))
        return

    uri = env_value("NEO4J_URI", env_file) or "bolt://localhost:7687"
    user = env_value("NEO4J_USER", env_file) or "neo4j"
    password = env_value("NEO4J_PASSWORD", env_file) or "password"
    with neo4j.GraphDatabase.driver(uri, auth=(user, password)) as driver:
        for name, label, prop, dimensions in vector:
            create_vector_index(
                driver,
                name,
                label=label,
                embedding_property=prop,
                dimensions=dimensions,
                similarity_fn="cosine",
                fail_if_exists=False,
            )
            print(colour(f"  created vector index {name}", GREEN))
        for name, label, prop in fulltext:
            create_fulltext_index(
                driver, name, label=label, node_properties=[prop], fail_if_exists=False
            )
            print(colour(f"  created fulltext index {name}", GREEN))


def configure_provider(provider: Provider, assume_yes: bool) -> None:
    """Prompt for, validate and store one provider's key."""
    env_file = read_env_file()
    name = provider.env_vars[0]
    existing = env_value(name, env_file)

    print()
    print(colour(f"-- {provider.label}", BOLD))
    if provider.free_tier:
        print(f"   {provider.free_tier}")
    if provider.manual:
        print(colour(f"   note: {provider.manual}", YELLOW))

    if existing:
        print(f"   {name} is already set ({mask(existing)})")
        if provider.validator:
            ok, detail = provider.validator(existing)
            print(
                f"   validation: {colour('ok', GREEN) if ok else colour(detail, RED)}"
            )
            if ok and not confirm("   Replace it anyway?", False, assume_yes):
                return
        elif not confirm("   Replace it?", False, assume_yes):
            return
    elif not confirm(f"   Set up {provider.label} now?", True, assume_yes):
        return

    if provider.signup_url:
        print(f"   get a key at: {provider.signup_url}")
        if not assume_yes and sys.platform == "darwin":
            if confirm("   Open that page in your browser?", True, assume_yes):
                subprocess.run(["open", provider.signup_url])

    if assume_yes:
        print("   non-interactive: cannot prompt for a key, skipping")
        return

    # No echo: the key must not land in scrollback or a screen recording.
    key = getpass(f"   paste {name} (input hidden, blank to skip): ").strip()
    if not key:
        print("   skipped")
        return

    if provider.validator:
        ok, detail = provider.validator(key)
        if not ok:
            # An invalid key is never written - a bad value in .env is worse
            # than none, because it turns a clear "unset" into a confusing 401.
            print(colour(f"   rejected: {detail} - not written to .env", RED))
            return
        print(colour("   validated", GREEN))

    write_env_var(name, key)
    print(colour(f"   wrote {name}={mask(key)} to .env (mode 0600)", GREEN))


def tier_1_keys(assume_yes: bool, only: Optional[str]) -> None:
    heading(f"Tier 1: {TIER_NAMES[1]}")
    print("Free without a credit card: Gemini, Cohere, Mistral.")
    for provider in PROVIDERS.values():
        if provider.tier != 1 or not provider.env_vars:
            continue
        if only and provider.key != only:
            continue
        configure_provider(provider, assume_yes)
