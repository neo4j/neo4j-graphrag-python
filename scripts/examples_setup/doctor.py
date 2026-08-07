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
"""Report what is missing, and change nothing."""

from __future__ import annotations

import sys
from dataclasses import dataclass

import example_requirements as reqs

from . import ENV_FILE
from .console import BOLD, DIM, GREEN, RED, YELLOW, colour
from .envfile import apply_env_file, env_value, read_env_file
from .probes import (
    Neo4jState,
    command_exists,
    extra_installed,
    ollama_models,
    package_installed,
    probe_neo4j,
)
from .providers import OLLAMA_CHAT_MODEL, PROVIDERS


@dataclass
class Blocker:
    """Why one example cannot run right now."""

    reason: str
    fix: str


def blockers_for(
    requirement: reqs.ExampleRequirements,
    env_file: dict[str, str],
    neo4j_state: Neo4jState,
) -> list[Blocker]:
    found: list[Blocker] = []

    for extra in sorted(requirement.extras):
        if not extra_installed(extra):
            found.append(
                Blocker(f"extra '{extra}' not installed", "uv sync --all-extras")
            )

    for module in sorted(requirement.install_hints()):
        if not package_installed(module.replace("-", "_")):
            found.append(Blocker(f"package '{module}' missing", f"uv add {module}"))

    for provider_key in sorted(requirement.providers):
        provider = PROVIDERS.get(provider_key)
        if provider is None:
            continue
        for name in provider.env_vars:
            if not env_value(name, env_file):
                found.append(
                    Blocker(
                        f"{name} not set",
                        f"setup_examples.py --provider {provider_key}",
                    )
                )
        if provider.credential_probe is not None:
            ok, fix = provider.credential_probe()
            if not ok:
                found.append(Blocker(f"{provider.label}: no usable credentials", fix))

    for service in sorted(requirement.services):
        if service in {reqs.APOC}:
            continue
        if not reqs.service_available(service):
            found.append(
                Blocker(
                    f"{reqs.SERVICE_LABELS.get(service, service)} unreachable",
                    _service_fix(service),
                )
            )

    if reqs.APOC in requirement.services and neo4j_state.authenticated:
        if not neo4j_state.has_apoc:
            found.append(
                Blocker(
                    "APOC not installed in Neo4j", "use examples/docker-compose.yml"
                )
            )

    if requirement.indexes and reqs.NEO4J_LOCAL in requirement.services:
        if neo4j_state.authenticated:
            missing = sorted(set(requirement.indexes) - neo4j_state.indexes)
            for name in missing:
                found.append(
                    Blocker(
                        f"index '{name}' does not exist",
                        "setup_examples.py --tier 0",
                    )
                )

    if requirement.sibling_modules:
        names = ", ".join(sorted(requirement.sibling_modules))
        found.append(
            Blocker(
                f"imports {names} from examples/data, which is not on sys.path",
                "run it as PYTHONPATH=examples/data python <file>",
            )
        )

    if requirement.uses_placeholder_model:
        if "ollama" in requirement.providers:
            found.append(
                Blocker(
                    "model name is the literal placeholder <model_name>",
                    f"edit the file to use e.g. {OLLAMA_CHAT_MODEL}",
                )
            )
        else:
            found.append(
                Blocker("model name is the literal placeholder <model_name>", "edit it")
            )

    if reqs.OLLAMA_SERVER in requirement.services and reqs.service_available(
        reqs.OLLAMA_SERVER
    ):
        pulled = ollama_models()
        if not pulled:
            found.append(
                Blocker(
                    "Ollama is running but no models are pulled",
                    f"ollama pull {OLLAMA_CHAT_MODEL}",
                )
            )

    return found


def _service_fix(service: str) -> str:
    if service in {
        reqs.NEO4J_LOCAL,
        reqs.WEAVIATE_SERVER,
        reqs.QDRANT_SERVER,
        reqs.PINECONE_SERVER,
    }:
        profile = " --profile vectordb" if service != reqs.NEO4J_LOCAL else ""
        return f"docker compose -f examples/docker-compose.yml{profile} up -d"
    if service == reqs.OLLAMA_SERVER:
        return "ollama serve"
    if service in {reqs.NEO4J_DEMO, reqs.INTERNET}:
        return "check your network connection"
    return ""


def run_doctor(strict: bool, verbose: bool) -> int:
    env_file = read_env_file()
    apply_env_file(env_file)
    neo4j_state = probe_neo4j(env_file)

    print(colour("Environment", BOLD))
    print(f"  python           {sys.version.split()[0]}")
    print(f"  .env             {'present' if ENV_FILE.exists() else 'absent'}")
    for tool in ("uv", "docker", "ollama", "gcloud", "aws", "az", "omni"):
        mark = colour("yes", GREEN) if command_exists(tool) else colour("no", DIM)
        print(f"  {tool:<16} {mark}")

    print()
    print(colour("Services", BOLD))
    service_rows = [
        reqs.NEO4J_LOCAL,
        reqs.NEO4J_DEMO,
        reqs.OLLAMA_SERVER,
        reqs.WEAVIATE_SERVER,
        reqs.QDRANT_SERVER,
        reqs.PINECONE_SERVER,
    ]
    for service in service_rows:
        up = reqs.service_available(service)
        mark = colour("up", GREEN) if up else colour("down", RED)
        print(f"  {service:<16} {mark:<16} {reqs.SERVICE_LABELS[service]}")
    if neo4j_state.reachable:
        if neo4j_state.authenticated:
            apoc = colour("yes", GREEN) if neo4j_state.has_apoc else colour("no", RED)
            print(f"  {'apoc':<16} {apoc}")
            print(f"  {'indexes':<16} {len(neo4j_state.indexes)} present")
        else:
            print(f"  {'neo4j auth':<16} {colour('failed', RED)} {neo4j_state.error}")

    print()
    print(colour("Credentials", BOLD))
    for provider in sorted(PROVIDERS.values(), key=lambda p: (p.tier, p.key)):
        if not provider.env_vars and provider.credential_probe is None:
            continue
        states = []
        for name in provider.env_vars:
            value = env_value(name, env_file)
            # Presence only. The value never reaches the terminal.
            states.append(
                f"{name}={colour('set', GREEN) if value else colour('unset', DIM)}"
            )
        if provider.credential_probe is not None:
            ok, _ = provider.credential_probe()
            states.append(
                f"credentials={colour('ok', GREEN) if ok else colour('none', DIM)}"
            )
        print(f"  tier {provider.tier}  {provider.label:<28} {'  '.join(states)}")

    print()
    print(colour("Python extras", BOLD))
    used_extras = sorted({e for r in reqs.analyse_all() for e in r.extras})
    for extra in used_extras:
        installed = extra_installed(extra)
        mark = colour("installed", GREEN) if installed else colour("missing", RED)
        print(f"  {extra:<24} {mark}")

    requirements = [r for r in reqs.analyse_all() if r.runnable]
    results: list[tuple[reqs.ExampleRequirements, list[Blocker]]] = [
        (r, blockers_for(r, env_file, neo4j_state)) for r in requirements
    ]
    ready = [r for r, b in results if not b]
    blocked = [(r, b) for r, b in results if b]

    print()
    print(colour("Examples", BOLD))
    print(
        f"  {colour(str(len(ready)), GREEN)} runnable, "
        f"{colour(str(len(blocked)), YELLOW)} blocked, "
        f"{len(reqs.analyse_all()) - len(requirements)} snippets with nothing to run"
    )

    if blocked:
        by_reason: dict[str, list[str]] = {}
        for requirement, found in blocked:
            key = f"{found[0].reason}  ->  {found[0].fix}"
            by_reason.setdefault(key, []).append(requirement.rel)
        print()
        print(colour("  Blocked, grouped by the first thing missing", BOLD))
        for reason in sorted(by_reason, key=lambda k: -len(by_reason[k])):
            paths = by_reason[reason]
            print(f"\n  {colour(reason, YELLOW)}  ({len(paths)} examples)")
            shown = paths if verbose else paths[:4]
            for path in shown:
                print(f"      {path}")
            if len(paths) > len(shown):
                print("      " + colour(f"... and {len(paths) - len(shown)} more", DIM))

    notes = [(r.rel, note) for r in reqs.analyse_all() for note in r.notes]
    if notes and verbose:
        print()
        print(colour("Known issues in the examples themselves", BOLD))
        for path, note in notes:
            print(f"  {path}\n      {note}")
    elif notes:
        print()
        print(
            colour(
                f"{len(notes)} examples carry known content issues; "
                "re-run with --verbose to list them.",
                DIM,
            )
        )

    if ready:
        print()
        print(colour("Ready to run, for example:", BOLD))
        for requirement in ready[:5]:
            print(f"  python {requirement.rel}")

    return 1 if (strict and blocked) else 0
