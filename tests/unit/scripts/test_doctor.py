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
"""Unit tests for the per-example blocker report.

`blockers_for` reaches the outside world through four module-level names in
`doctor` - `extra_installed`, `package_installed`, `env_value` and
`reqs.service_available` - so every one of these runs with those patched. No
network, no services, no credentials.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import example_requirements as reqs
import pytest
from examples_setup import doctor
from examples_setup.probes import Neo4jState

FAKE = Path(reqs.EXAMPLES_DIR) / "fake_example.py"


@pytest.fixture(autouse=True)
def everything_satisfied(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Default to a machine where nothing is missing.

    Each test then breaks exactly one thing, so a blocker in the result can only
    have come from the dimension under test.
    """
    monkeypatch.setattr(doctor, "extra_installed", lambda extra: True)
    monkeypatch.setattr(doctor, "package_installed", lambda module: True)
    monkeypatch.setattr(doctor, "env_value", lambda name, env_file: "set")
    monkeypatch.setattr(reqs, "service_available", lambda service: True)
    monkeypatch.setattr(
        doctor, "ollama_models", lambda: ["llama3.2", "nomic-embed-text"]
    )
    yield


def healthy_neo4j() -> Neo4jState:
    return Neo4jState(
        reachable=True,
        authenticated=True,
        has_apoc=True,
        indexes={"moviePlotsEmbedding", "movieFulltext"},
    )


def reasons(requirement: reqs.ExampleRequirements, state: Neo4jState) -> list[str]:
    return [b.reason for b in doctor.blockers_for(requirement, {}, state)]


def test_a_satisfied_example_has_no_blockers() -> None:
    requirement = reqs.ExampleRequirements(
        path=FAKE, extras={"openai"}, providers={"openai"}, services={reqs.NEO4J_LOCAL}
    )
    assert reasons(requirement, healthy_neo4j()) == []


def test_a_missing_extra_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(doctor, "extra_installed", lambda extra: False)
    requirement = reqs.ExampleRequirements(path=FAKE, extras={"openai"})
    assert reasons(requirement, healthy_neo4j()) == ["extra 'openai' not installed"]


def test_an_unset_env_var_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(doctor, "env_value", lambda name, env_file: None)
    requirement = reqs.ExampleRequirements(path=FAKE, providers={"openai"})
    found = reasons(requirement, healthy_neo4j())
    assert found == ["OPENAI_API_KEY not set"]


def test_the_env_var_fix_points_at_the_installer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Now that the installer ships, the doctor can hand the user straight to it.

    Every fix string must name a command that exists - before the installer
    landed this advised `--provider`, which nothing implemented.
    """
    monkeypatch.setattr(doctor, "env_value", lambda name, env_file: None)
    requirement = reqs.ExampleRequirements(path=FAKE, providers={"openai"})
    fixes = [b.fix for b in doctor.blockers_for(requirement, {}, healthy_neo4j())]
    assert "setup_examples.py --provider openai" in " ".join(fixes)


def test_an_unreachable_service_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(reqs, "service_available", lambda service: False)
    requirement = reqs.ExampleRequirements(path=FAKE, services={reqs.NEO4J_LOCAL})
    found = reasons(requirement, Neo4jState())
    assert any("unreachable" in reason for reason in found)


# ---------------------------------------------------------------------------
# Neo4j reachable but not authenticated
# ---------------------------------------------------------------------------


def test_a_rejected_neo4j_login_is_reported() -> None:
    """Regression: a wrong password used to produce a clean bill of health.

    service_available only opens a TCP socket, and the APOC and index checks both
    gate on `authenticated`, so every check silently passed while every Neo4j
    example failed at runtime.
    """
    state = Neo4jState(reachable=True, authenticated=False, error="unauthorized")
    requirement = reqs.ExampleRequirements(
        path=FAKE,
        services={reqs.NEO4J_LOCAL, reqs.APOC},
        indexes={"moviePlotsEmbedding"},
    )
    found = reasons(requirement, state)
    assert found, "a reachable but unauthenticated Neo4j must be reported"
    assert any("rejected the connection" in reason for reason in found)


def test_a_healthy_neo4j_does_not_report_a_login_problem() -> None:
    requirement = reqs.ExampleRequirements(path=FAKE, services={reqs.NEO4J_LOCAL})
    assert reasons(requirement, healthy_neo4j()) == []


# ---------------------------------------------------------------------------
# Neo4j contents
# ---------------------------------------------------------------------------


def test_missing_apoc_is_reported() -> None:
    state = Neo4jState(reachable=True, authenticated=True, has_apoc=False)
    requirement = reqs.ExampleRequirements(
        path=FAKE, services={reqs.NEO4J_LOCAL, reqs.APOC}
    )
    assert "APOC not installed in Neo4j" in reasons(requirement, state)


def test_a_missing_index_is_reported_without_naming_a_dead_flag() -> None:
    state = Neo4jState(reachable=True, authenticated=True, has_apoc=True, indexes=set())
    requirement = reqs.ExampleRequirements(
        path=FAKE, services={reqs.NEO4J_LOCAL}, indexes={"moviePlotsEmbedding"}
    )
    blockers = doctor.blockers_for(requirement, {}, state)
    assert any("does not exist" in b.reason for b in blockers)
    assert "--tier" not in " ".join(b.fix for b in blockers)


# ---------------------------------------------------------------------------
# Other dimensions
# ---------------------------------------------------------------------------


def test_a_sibling_module_import_is_reported() -> None:
    requirement = reqs.ExampleRequirements(
        path=FAKE, sibling_modules={"embedding_avatar"}
    )
    found = reasons(requirement, healthy_neo4j())
    assert any("examples/data" in reason for reason in found)


@pytest.mark.parametrize(
    "pulled,expected",
    [(["llama3.2"], 1), ([], 2), (["llama3.2", "nomic-embed-text"], 0)],
)
def test_ollama_reports_each_model_the_examples_need(
    monkeypatch: pytest.MonkeyPatch, pulled: list[str], expected: int
) -> None:
    """The chat and embedding examples need different models."""
    monkeypatch.setattr(doctor, "ollama_models", lambda: pulled)
    requirement = reqs.ExampleRequirements(path=FAKE, services={reqs.OLLAMA_SERVER})
    found = [r for r in reasons(requirement, healthy_neo4j()) if "Ollama" in r]
    assert len(found) == expected


def test_azure_reports_a_blocker_rather_than_looking_ready() -> None:
    """Its endpoint and key are hardcoded placeholders, not env vars.

    With no env vars declared and no probe there is nothing to detect, so the
    example would be reported ready and then fail the moment it ran.
    """
    requirement = reqs.ExampleRequirements(path=FAKE, providers={"azure"})
    found = reasons(requirement, healthy_neo4j())
    assert any("Azure" in reason for reason in found)
