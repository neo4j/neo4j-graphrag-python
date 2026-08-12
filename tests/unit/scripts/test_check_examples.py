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
"""Unit tests for the examples static checker.

The rules take a parsed module, so every case here is a source string. Nothing
touches the network, an API key, or the filesystem beyond reading paths that
already exist in the repository.
"""

from __future__ import annotations

import ast
from pathlib import Path

import check_examples
import example_requirements as reqs
import pytest
from examples_setup.doctor import Blocker
from examples_setup.probes import Neo4jState
from check_examples import (
    MIN_REASONING_BUDGET,
    check_data_files,
    check_llm_usage,
    dict_literal_bindings,
)

FAKE_PATH = Path("examples/fake_example.py")


def messages(source: str) -> list[str]:
    """The problem messages the LLM rules report for a snippet."""
    return [p.message for p in check_llm_usage(FAKE_PATH, ast.parse(source))]


# ---------------------------------------------------------------------------
# model_params resolution
# ---------------------------------------------------------------------------


def test_model_params_bound_to_a_variable_is_resolved() -> None:
    """The rules must see params built into a local first.

    Regression: reading only the call site meant `model_params=<name>` resolved
    to an empty dict and every rule below silently skipped.
    """
    source = """
params = {"temperature": 0, "max_completion_tokens": 2000}
llm = OpenAILLM(model_name="gpt-5", model_params=params)
"""
    assert len(messages(source)) == 2


def test_model_params_as_a_dict_literal_is_still_resolved() -> None:
    source = """
llm = OpenAILLM(model_name="gpt-5", model_params={"temperature": 0})
"""
    assert len(messages(source)) == 1


def test_a_name_bound_to_two_different_dicts_is_not_guessed_at() -> None:
    """The walk is not scope-aware, so an ambiguous name must be left alone."""
    source = """
def a() -> None:
    params = {"temperature": 0}
    OpenAILLM(model_name="gpt-5", model_params=params)

def b() -> None:
    params = {"reasoning_effort": "low"}
    OpenAILLM(model_name="gpt-5", model_params=params)
"""
    assert messages(source) == []
    assert "params" not in dict_literal_bindings(ast.parse(source))


def test_a_name_bound_to_the_same_dict_twice_is_not_ambiguous() -> None:
    source = """
params = {"temperature": 0}
params = {"temperature": 0}
"""
    assert dict_literal_bindings(ast.parse(source)) == {"params": {"temperature": 0}}


# ---------------------------------------------------------------------------
# Reasoning-model rules
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "params,expected",
    [
        ('{"max_tokens": 2000}', 1),
        (f'{{"max_completion_tokens": {MIN_REASONING_BUDGET}}}', 0),
        (f'{{"max_completion_tokens": {MIN_REASONING_BUDGET - 1}}}', 1),
        ('{"temperature": 0}', 1),
        ('{"temperature": 1}', 0),
        ('{"reasoning_effort": "low"}', 0),
    ],
)
def test_reasoning_model_params(params: str, expected: int) -> None:
    source = f'OpenAILLM(model_name="gpt-5", model_params={params})'
    assert len(messages(source)) == expected


def test_non_reasoning_model_is_left_alone() -> None:
    """gpt-4.1 accepts both of the parameters gpt-5 rejects."""
    source = """
OpenAILLM(model_name="gpt-4.1", model_params={"max_tokens": 100, "temperature": 0})
"""
    assert messages(source) == []


def test_placeholder_model_is_not_judged() -> None:
    source = 'OpenAILLM(model_name="<model_name>", model_params={"temperature": 0})'
    assert messages(source) == []


# ---------------------------------------------------------------------------
# LangChain's client, which sends a temperature of its own
# ---------------------------------------------------------------------------


def test_langchain_client_without_a_temperature_is_flagged() -> None:
    """ChatOpenAI sends its own default, so omitting one does not help."""
    assert len(messages('ChatOpenAI(model="gpt-5")')) == 1


def test_langchain_client_with_the_only_accepted_temperature_is_not_flagged() -> None:
    """Regression: a valid temperature=1 was reported with 'pass temperature=1'."""
    assert messages('ChatOpenAI(model="gpt-5", temperature=1)') == []


def test_langchain_client_with_a_rejected_temperature_is_flagged_once() -> None:
    assert len(messages('ChatOpenAI(model="gpt-5", temperature=0)')) == 1


# ---------------------------------------------------------------------------
# Other providers: the false-positive guards
# ---------------------------------------------------------------------------


def test_anthropic_max_tokens_is_not_flagged() -> None:
    """max_tokens is required by AnthropicLLM - it must not be 'fixed'."""
    source = 'AnthropicLLM(model_name="claude-sonnet-4-5", model_params={"max_tokens": 1000})'
    assert messages(source) == []


def test_ollama_nested_options_are_not_flagged() -> None:
    """Ollama nests generation params under 'options', so the keys differ."""
    source = """
OllamaLLM(model_name="llama3.2", model_params={"options": {"temperature": 0}})
"""
    assert messages(source) == []


# ---------------------------------------------------------------------------
# temperature passed to invoke() rather than the constructor
# ---------------------------------------------------------------------------


def test_temperature_on_invoke_is_attributed_to_the_model() -> None:
    source = """
llm = OpenAILLM(model_name="gpt-5")
llm.invoke("hello", temperature=0)
"""
    assert len(messages(source)) == 1


def test_temperature_on_invoke_of_a_non_reasoning_model_is_fine() -> None:
    source = """
llm = OpenAILLM(model_name="gpt-4.1")
llm.invoke("hello", temperature=0)
"""
    assert messages(source) == []


# ---------------------------------------------------------------------------
# Data files
# ---------------------------------------------------------------------------


def test_missing_data_file_is_flagged() -> None:
    source = 'path = "examples/data/does_not_exist.pdf"'
    problems = check_data_files(FAKE_PATH, ast.parse(source))
    assert len(problems) == 1
    assert "does_not_exist.pdf" in problems[0].message


def test_existing_data_file_is_not_flagged() -> None:
    source = (
        'path = "examples/data/Harry Potter and the Chamber of Secrets Summary.pdf"'
    )
    assert check_data_files(FAKE_PATH, ast.parse(source)) == []


@pytest.mark.parametrize(
    "value",
    [
        "https://example.com/remote.pdf",  # not a local path
        "<path_to>/file.pdf",  # a docs placeholder
        "bare_name.pdf",  # joined to a directory elsewhere
    ],
)
def test_data_file_values_that_are_not_judged(value: str) -> None:
    assert check_data_files(FAKE_PATH, ast.parse(f'path = "{value}"')) == []


# ---------------------------------------------------------------------------
# --live gating
#
# What "runnable right now" means is the doctor's judgement; these pin that the
# runner asks it, rather than keeping a second opinion of its own.
# ---------------------------------------------------------------------------


def test_snippets_are_never_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """blockers_for returns [] for a snippet, so the runner keeps its own gate.

    Without it, every file with nothing to run would be executed.
    """
    ran: list[Path] = []
    monkeypatch.setattr(check_examples, "blockers_for", lambda r, e, n: [])
    monkeypatch.setattr(check_examples, "read_env_file", lambda: {})
    monkeypatch.setattr(check_examples, "apply_env_file", lambda env: None)
    monkeypatch.setattr(check_examples, "probe_neo4j", lambda env: Neo4jState())
    monkeypatch.setattr(check_examples, "_confirm_spend", lambda ready, yes: False)

    def spy(cmd: list[str], **kwargs: object) -> None:
        ran.append(Path(cmd[1]))

    monkeypatch.setattr("check_examples.subprocess.run", spy)
    check_examples.run_live_checks(timeout=1, assume_yes=True)
    assert ran == [], "declining the spend must run nothing"


def test_a_blocked_example_is_skipped_not_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(check_examples, "read_env_file", lambda: {})
    monkeypatch.setattr(check_examples, "apply_env_file", lambda env: None)
    monkeypatch.setattr(check_examples, "probe_neo4j", lambda env: Neo4jState())
    monkeypatch.setattr(
        check_examples,
        "blockers_for",
        lambda r, e, n: [Blocker("OPENAI_API_KEY not set", "export it")],
    )
    ran: list[str] = []
    monkeypatch.setattr(
        "check_examples.subprocess.run", lambda cmd, **kw: ran.append(cmd[1])
    )
    rc = check_examples.run_live_checks(timeout=1, assume_yes=True)
    assert ran == []
    assert rc == 0


def test_declining_the_cost_prompt_runs_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(check_examples, "read_env_file", lambda: {})
    monkeypatch.setattr(check_examples, "apply_env_file", lambda env: None)
    monkeypatch.setattr(check_examples, "probe_neo4j", lambda env: Neo4jState())
    monkeypatch.setattr(check_examples, "blockers_for", lambda r, e, n: [])
    monkeypatch.setattr("builtins.input", lambda prompt="": "n")
    ran: list[str] = []
    monkeypatch.setattr(
        "check_examples.subprocess.run", lambda cmd, **kw: ran.append(cmd[1])
    )
    rc = check_examples.run_live_checks(timeout=1, assume_yes=False)
    assert ran == []
    assert rc == 0


def test_the_cost_prompt_names_the_providers_that_will_be_billed(
    capsys: pytest.CaptureFixture[str],
) -> None:
    paid = reqs.EXAMPLES_DIR / "customize" / "llms" / "anthropic_llm.py"
    check_examples._confirm_spend([paid], assume_yes=True)
    out = capsys.readouterr().out
    assert "COSTS MONEY" in out
    assert "anthropic" in out


def test_azure_is_not_considered_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    """Its credentials are hardcoded placeholders, not environment variables.

    The provider declares no env vars, so without a probe the doctor reports it
    ready and --live runs it, turning a clean SKIP into a FAIL.
    """
    from examples_setup.doctor import blockers_for as real_blockers_for

    requirement = reqs.ExampleRequirements(
        path=reqs.EXAMPLES_DIR / "fake.py", providers={"azure"}
    )
    found = real_blockers_for(requirement, {}, Neo4jState())
    assert found, "azure must report a blocker rather than looking ready"
