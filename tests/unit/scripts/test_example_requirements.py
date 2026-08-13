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
"""Unit tests for the model of what each example needs.

The scan is pure AST work over files already in the repository, so nothing here
touches the network or a service.
"""

from __future__ import annotations

import ast
import fnmatch
import tempfile
from pathlib import Path

import example_requirements as reqs
import pytest


def analyse_rel(rel: str) -> reqs.ExampleRequirements:
    return reqs.analyse(reqs.EXAMPLES_DIR / rel)


# ---------------------------------------------------------------------------
# Requirements declared in a config file rather than in the imports
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rel",
    [
        "build_graph/from_config_files/simple_kg_pipeline_from_config_file.py",
        "build_graph/from_config_files/simple_kg_pipeline_from_config_file_with_url.py",
        "customize/build_graph/pipeline/from_config_files/pipeline_from_config_file.py",
    ],
)
def test_config_file_examples_still_need_openai(rel: str) -> None:
    """Their LLM is named in YAML/JSON, which the import scan cannot see.

    Regression: these three were reported as needing nothing at all, so the
    doctor counted them runnable with no key and no openai extra.
    """
    requirement = analyse_rel(rel)
    assert "openai" in requirement.extras
    assert "openai" in requirement.providers
    assert "OPENAI_API_KEY" in requirement.env_vars


# ---------------------------------------------------------------------------
# Packages no extra provides
# ---------------------------------------------------------------------------


def analyse_source(source: str) -> reqs.ExampleRequirements:
    """Run the real analyse() over a synthetic file.

    Exercising the whole pipeline rather than hand-building an
    ExampleRequirements is the point: a hand-built object cannot tell you whether
    analyse() populated it correctly.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "synthetic_example.py"
        path.write_text(source, encoding="utf-8")
        return reqs.analyse(path)


def test_analyse_keeps_a_package_that_no_extra_declares() -> None:
    """Regression: install_hints() could only ever return [].

    analyse() narrowed modules to the ones already known, which is precisely the
    set install_hints() then filters back out - so asserting the corpus reports
    nothing passes either way. This pins the fixed behaviour directly.
    """
    requirement = analyse_source("import cv2\n")
    assert "cv2" in requirement.modules
    assert requirement.install_hints() == ["cv2"]


def test_no_example_imports_a_package_nothing_declares() -> None:
    """The corpus invariant, now that the mechanism above is pinned separately."""
    hinted = {
        module
        for requirement in reqs.analyse_all()
        for module in requirement.install_hints()
    }
    assert hinted == set(), (
        f"examples import packages no extra declares: {sorted(hinted)}. "
        "Declare them in pyproject.toml, or map them in MODULE_EXTRAS."
    )


def test_install_hints_reports_the_import_name() -> None:
    requirement = reqs.ExampleRequirements(
        path=reqs.EXAMPLES_DIR / "fake.py", modules={"cv2"}
    )
    assert requirement.install_hints() == ["cv2"]


def test_core_dependencies_are_never_hinted() -> None:
    """pydantic and friends ship with the library itself."""
    requirement = reqs.ExampleRequirements(
        path=reqs.EXAMPLES_DIR / "fake.py",
        modules={m for m in reqs.ALWAYS_INSTALLED},
    )
    assert requirement.install_hints() == []


# ---------------------------------------------------------------------------
# The import scan
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "source,expected_module,expected_symbol",
    [
        ("import openai", "openai", None),
        ("import openai as oai", "openai", None),
        ("from openai import OpenAI", "openai", "OpenAI"),
        ("from neo4j_graphrag.llm import OpenAILLM", "neo4j_graphrag", "OpenAILLM"),
        (
            "from neo4j_graphrag.llm import OpenAILLM as LLM",
            "neo4j_graphrag",
            "OpenAILLM",
        ),
        ("def f():\n    import cohere", "cohere", None),
        ("if True:\n    import cohere", "cohere", None),
    ],
)
def test_imported_names(
    source: str, expected_module: str, expected_symbol: str | None
) -> None:
    modules, symbols = reqs._imported_names(ast.parse(source))
    assert expected_module in modules
    if expected_symbol is not None:
        assert expected_symbol in symbols


def test_relative_imports_are_not_third_party() -> None:
    modules, _ = reqs._imported_names(ast.parse("from .thing import Thing"))
    assert modules == set()


def test_an_aliased_symbol_still_maps_to_its_extra() -> None:
    """The extra follows the real name, not what the example called it."""
    requirement = analyse_source("from neo4j_graphrag.llm import OpenAILLM as LLM\n")
    assert requirement.extras == {"openai"}
    assert requirement.providers == {"openai"}


# ---------------------------------------------------------------------------
# Env vars and services
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "source,expected",
    [
        ('os.getenv("OPENAI_API_KEY")', {"OPENAI_API_KEY"}),
        ('os.environ["NEO4J_URI"]', {"NEO4J_URI"}),
        ('os.environ.get("COHERE_API_KEY")', {"COHERE_API_KEY"}),
        ("os.getenv(name)", set()),
    ],
)
def test_env_vars(source: str, expected: set[str]) -> None:
    assert reqs._env_vars(ast.parse(source)) == expected


def test_a_local_uri_means_a_local_neo4j() -> None:
    services = reqs._datastore_services(["bolt://localhost:7687"], set(), mocked=False)
    assert reqs.NEO4J_LOCAL in services


def test_the_demo_host_means_the_demo_database() -> None:
    services = reqs._datastore_services(
        [f"neo4j+s://{reqs._DEMO_HOST}"], set(), mocked=False
    )
    assert reqs.NEO4J_DEMO in services


def test_a_mocked_driver_needs_no_service() -> None:
    assert (
        reqs._datastore_services(["bolt://localhost:7687"], set(), mocked=True) == set()
    )


# ---------------------------------------------------------------------------
# Whole-corpus invariants
# ---------------------------------------------------------------------------


def test_every_example_is_analysable() -> None:
    """analyse() promises never to raise on a file in examples/."""
    unparsed = [
        r.rel
        for r in reqs.analyse_all()
        if not r.runnable and "does not parse" in r.notes
    ]
    assert unparsed == []


def test_service_rule_globs_all_match_something() -> None:
    """A glob that matches nothing is a rule silently doing nothing."""
    paths = [
        p.relative_to(reqs.EXAMPLES_DIR).as_posix() for p in reqs.iter_example_files()
    ]
    unmatched = [
        pattern
        for pattern, _ in reqs.SERVICE_RULES
        if not any(fnmatch.fnmatch(path, pattern) for path in paths)
    ]
    assert unmatched == []


# ---------------------------------------------------------------------------
# Table invariants
#
# Each of these is one line, and each closes a way for a half-finished edit to
# produce a plausible wrong answer instead of an error.
# ---------------------------------------------------------------------------


def test_every_symbol_maps_to_both_an_extra_and_a_provider() -> None:
    """The two tables are halves of one fact and must stay in step.

    Their *values* legitimately differ - AzureOpenAILLM needs the `openai` extra
    but talks to the `azure` provider - so they cannot be merged. Their keys
    cannot differ: a symbol in one but not the other means the doctor either
    installs a package it never asks for a key for, or the reverse.
    """
    assert set(reqs.SYMBOL_EXTRAS) == set(reqs.SYMBOL_PROVIDERS)


def test_every_module_provider_also_declares_an_extra() -> None:
    """Subset, not equality: most modules imply an extra without implying a
    provider, but a module that reaches a provider must also install something.
    """
    assert set(reqs.MODULE_PROVIDERS) <= set(reqs.MODULE_EXTRAS)


def test_every_service_an_example_needs_has_a_label() -> None:
    """service_available() returns True for anything it cannot probe, so a
    typo'd service name would silently report as satisfied.
    """
    seen = {s for r in reqs.analyse_all() for s in r.services}
    assert seen <= set(reqs.SERVICE_LABELS)


# How many files each rule is expected to match. A bare "matches something"
# assertion only protects the rules that match exactly one file; renaming one of
# several matches leaves the pattern non-empty and the requirement silently lost.
# Pinning the count also catches the opposite error - a glob widened until it
# sweeps in a file that does not want the requirement.
#
# Update deliberately when examples/ gains or loses a file: that is exactly the
# moment to confirm the rule still applies to what it now matches.
EXPECTED_RULE_MATCHES = {
    "kg_builder.py": 1,
    "build_graph/**": 6,
    "customize/build_graph/pipeline/kg_builder_*.py": 3,
    "customize/build_graph/pipeline/text_to_lexical_graph_to_entity_graph_*.py": 2,
    "build_graph/from_config_files/simple_kg_pipeline_from_config_file_with_url.py": 1,
    "customize/build_graph/components/loaders/pdf_loader_from_url.py": 1,
    "retrieve/tools/tools_retriever_example.py": 1,
    "retrieve/similarity_search_for_*.py": 2,
    "retrieve/vector_cypher_retriever.py": 1,
    "retrieve/hybrid_*.py": 2,
    "customize/retrievers/result_formatter_*.py": 2,
    "customize/retrievers/use_pre_filters.py": 1,
    "customize/answer/custom_prompt.py": 1,
    "customize/answer/langchain_compatiblity.py": 1,
    "customize/retrievers/external/weaviate/*.py": 3,
    "customize/retrievers/external/qdrant/*.py": 3,
    "customize/retrievers/external/pinecone/*.py": 2,
    "build_graph/from_config_files/simple_kg_pipeline_from_config_file*.py": 2,
    "customize/build_graph/pipeline/from_config_files/pipeline_from_config_file.py": 1,
}


def test_service_rules_match_the_files_they_are_meant_to() -> None:
    paths = [
        p.relative_to(reqs.EXAMPLES_DIR).as_posix() for p in reqs.iter_example_files()
    ]
    actual = {
        pattern: sum(1 for path in paths if fnmatch.fnmatch(path, pattern))
        for pattern, _ in reqs.SERVICE_RULES
    }
    assert actual == EXPECTED_RULE_MATCHES
