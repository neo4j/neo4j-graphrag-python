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


def test_install_hints_reports_a_package_behind_no_extra() -> None:
    """Regression: install_hints() could only ever return [].

    analyse() narrowed modules to the ones already known, which is precisely the
    set install_hints() then filtered back out.
    """
    hinted = {
        module
        for requirement in reqs.analyse_all()
        for module in requirement.install_hints()
    }
    assert hinted == set(), (
        f"examples import packages no extra declares: {sorted(hinted)}. "
        "Declare them in pyproject.toml, or map them in MODULE_EXTRAS."
    )


def test_install_hints_names_the_distribution_not_the_import() -> None:
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
    requirement = reqs.ExampleRequirements(path=reqs.EXAMPLES_DIR / "fake.py")
    _, symbols = reqs._imported_names(
        ast.parse("from neo4j_graphrag.llm import OpenAILLM as LLM")
    )
    assert "OpenAILLM" in symbols
    assert reqs.SYMBOL_EXTRAS["OpenAILLM"] == "openai"
    assert requirement.extras == set()


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
    import fnmatch

    unmatched = [
        pattern
        for pattern, _ in reqs.SERVICE_RULES
        if not any(fnmatch.fnmatch(path, pattern) for path in paths)
    ]
    assert unmatched == []
