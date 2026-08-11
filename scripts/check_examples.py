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
"""Static checks for the ``examples/`` directory.

The examples are the project's shop window but no CI job exercises them, so
model-reference drift lands in ``main`` unnoticed. A bulk model swap in #482
left 15 call sites passing parameters their new model rejects, and the failures
were deterministic - nobody had run them.

This script catches that class of breakage offline: no API key, no network, no
cost, fast enough for a pre-commit hook.

    python scripts/check_examples.py            # static checks (default)
    python scripts/check_examples.py --live     # additionally run what it can

What it checks:

1. Every example parses.
2. LLM parameters are valid for the model, per provider. Providers disagree
   about this - OpenAI reasoning models reject ``max_tokens`` while Anthropic
   requires it - so the rules are per-family, not global.
3. Referenced data files exist.

Known limitation: rule 2 resolves a model only through a simple name binding
(``llm = OpenAILLM(...)`` or ``with OpenAILLM(...) as llm``). A model reached
any other way - returned from a helper, stored on an object, picked from a dict
- is not attributed, and its ``invoke()`` kwargs go unchecked.
"""

from __future__ import annotations

import argparse
import ast
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

import example_requirements  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples"

# Reasoning models bill their internal reasoning against the completion budget.
# Below roughly this many tokens the budget is consumed before any visible
# output, and the response comes back empty with finish_reason="length".
MIN_REASONING_BUDGET = 8000

# Model-name prefixes whose families reject `max_tokens` and non-default
# `temperature`. Extend as new reasoning families ship.
REASONING_PREFIXES = ("gpt-5", "o1", "o3", "o4")

# Constructors that speak to OpenAI, so the reasoning rules apply to them.
OPENAI_CONSTRUCTORS = {"OpenAILLM", "AzureOpenAILLM"}

# LangChain's chat wrapper. Distinct from the above because it *always* sends a
# temperature (its own default, historically 0.7), so a reasoning model fails
# even when the caller passes no temperature at all.
LANGCHAIN_OPENAI_CONSTRUCTORS = {"ChatOpenAI"}

# Non-OpenAI providers. Listed so the checker can tell "known and intentionally
# unchecked" from "unrecognised", and as the place to hang per-provider rules.
OTHER_PROVIDER_CONSTRUCTORS = {
    "AnthropicLLM",  # max_tokens is required here - do not "fix" it
    "BedrockLLM",  # camelCase maxTokens
    "CohereLLM",
    "GeminiLLM",
    "MistralAILLM",
    "OllamaLLM",  # params nest under "options"
    "VertexAILLM",
}

ALL_LLM_CONSTRUCTORS = (
    OPENAI_CONSTRUCTORS | LANGCHAIN_OPENAI_CONSTRUCTORS | OTHER_PROVIDER_CONSTRUCTORS
)

INVOKE_METHODS = {
    "invoke",
    "ainvoke",
    "invoke_with_tools",
    "ainvoke_with_tools",
}

DATA_SUFFIXES = (".pdf", ".txt", ".csv")


@dataclass
class Problem:
    path: Path
    line: int
    message: str
    fix: str

    def render(self) -> str:
        rel = self.path.relative_to(REPO_ROOT)
        return f"{rel}:{self.line}: {self.message}\n    fix: {self.fix}"


@dataclass
class Binding:
    """A name bound to an LLM constructor call."""

    constructor: str
    model: Optional[str]
    params: dict[str, Any]
    line: int


@dataclass
class FileReport:
    path: Path
    problems: list[Problem] = field(default_factory=list)


def is_reasoning_model(model: Optional[str]) -> bool:
    return isinstance(model, str) and model.startswith(REASONING_PREFIXES)


def is_placeholder(model: Optional[str]) -> bool:
    """True for docs placeholders like "<model_name>", which never run."""
    return isinstance(model, str) and ("<" in model or ">" in model)


def literal_or_none(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return None


def constructor_name(node: ast.Call) -> Optional[str]:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def dict_literal_bindings(tree: ast.Module) -> dict[str, dict[str, Any]]:
    """Names bound to a dict literal, for `model_params=SOME_NAME`.

    Examples often build their parameters into a local first, which no amount of
    literal_eval on the call site can see. This walk is not scope-aware, so a name
    bound more than once to differing dicts is dropped rather than guessed at.
    """
    bindings: dict[str, dict[str, Any]] = {}
    ambiguous: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        value = literal_or_none(node.value)
        if not isinstance(value, dict):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue
            if target.id in bindings and bindings[target.id] != value:
                ambiguous.add(target.id)
            bindings[target.id] = value
    for name in ambiguous:
        del bindings[name]
    return bindings


def extract_model_and_params(
    node: ast.Call, dict_bindings: Optional[dict[str, dict[str, Any]]] = None
) -> tuple[Optional[str], dict[str, Any]]:
    model: Optional[str] = None
    params: dict[str, Any] = {}
    for kw in node.keywords:
        if kw.arg in ("model_name", "model"):
            value = literal_or_none(kw.value)
            if isinstance(value, str):
                model = value
        elif kw.arg == "model_params":
            value = literal_or_none(kw.value)
            if isinstance(value, dict):
                params = value
            elif isinstance(kw.value, ast.Name) and dict_bindings:
                params = dict_bindings.get(kw.value.id, {})
        elif kw.arg == "temperature":
            # ChatOpenAI and friends take temperature directly.
            params["temperature"] = literal_or_none(kw.value)
    return model, params


def check_openai_params(
    path: Path,
    line: int,
    model: Optional[str],
    params: dict[str, Any],
    *,
    always_sends_temperature: bool,
) -> list[Problem]:
    """Apply the OpenAI reasoning-model rules to one construction site."""
    if is_placeholder(model) or not is_reasoning_model(model):
        return []

    problems: list[Problem] = []

    if "max_tokens" in params:
        problems.append(
            Problem(
                path,
                line,
                f"{model!r} rejects 'max_tokens'",
                "rename it to 'max_completion_tokens'",
            )
        )

    temperature = params.get("temperature")
    if "temperature" in params and temperature != 1:
        problems.append(
            Problem(
                path,
                line,
                f"{model!r} only accepts the default temperature (1), got {temperature!r}",
                "drop 'temperature', or use a non-reasoning model such as gpt-4.1",
            )
        )
    elif always_sends_temperature and "temperature" not in params:
        # No explicit temperature, but this client sends its own default anyway.
        problems.append(
            Problem(
                path,
                line,
                f"{model!r} with a LangChain chat client, which always sends a "
                "temperature (default 0.7) that reasoning models reject",
                "use a non-reasoning model such as gpt-4.1, or pass temperature=1",
            )
        )

    budget = params.get("max_completion_tokens")
    if isinstance(budget, int) and budget < MIN_REASONING_BUDGET:
        problems.append(
            Problem(
                path,
                line,
                f"'max_completion_tokens' of {budget} is too small for {model!r}: "
                "reasoning tokens count against it and can consume the whole "
                "budget, returning empty content",
                f"raise it to at least {MIN_REASONING_BUDGET}",
            )
        )

    return problems


def check_llm_usage(path: Path, tree: ast.Module) -> list[Problem]:
    """Check construction sites, then invoke() kwargs on bound names."""
    problems: list[Problem] = []
    bindings: dict[str, Binding] = {}
    dicts = dict_literal_bindings(tree)

    def record(target: ast.expr, call: ast.Call) -> None:
        name = constructor_name(call)
        if name not in ALL_LLM_CONSTRUCTORS or not isinstance(target, ast.Name):
            return
        model, params = extract_model_and_params(call, dicts)
        bindings[target.id] = Binding(name, model, params, call.lineno)

    for node in ast.walk(tree):
        # Construction sites, whether or not the result is bound to a name.
        if isinstance(node, ast.Call):
            name = constructor_name(node)
            if name in OPENAI_CONSTRUCTORS or name in LANGCHAIN_OPENAI_CONSTRUCTORS:
                model, params = extract_model_and_params(node, dicts)
                problems.extend(
                    check_openai_params(
                        path,
                        node.lineno,
                        model,
                        params,
                        always_sends_temperature=name in LANGCHAIN_OPENAI_CONSTRUCTORS,
                    )
                )
        # Name bindings, so invoke() kwargs can be attributed to a model.
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(node.value, ast.Call):
                    record(target, node.value)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.value, ast.Call):
            record(node.target, node.value)
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                if item.optional_vars is not None and isinstance(
                    item.context_expr, ast.Call
                ):
                    record(item.optional_vars, item.context_expr)

    # invoke(..., temperature=...) on a name we resolved to a reasoning model.
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr not in INVOKE_METHODS:
            continue
        if not isinstance(func.value, ast.Name):
            continue
        binding = bindings.get(func.value.id)
        if binding is None or binding.constructor not in OPENAI_CONSTRUCTORS:
            continue
        if not is_reasoning_model(binding.model):
            continue
        for kw in node.keywords:
            if kw.arg != "temperature":
                continue
            value = literal_or_none(kw.value)
            if value != 1:
                problems.append(
                    Problem(
                        path,
                        node.lineno,
                        f"temperature={value!r} passed to {func.attr}() on "
                        f"{func.value.id!r}, which is {binding.model!r} "
                        f"(constructed at line {binding.line}) and only accepts "
                        "the default (1)",
                        "drop the temperature kwarg, or construct this LLM with "
                        "a non-reasoning model such as gpt-4.1",
                    )
                )

    return problems


def check_data_files(path: Path, tree: ast.Module) -> list[Problem]:
    """Flag literal data-file paths that do not resolve."""
    problems: list[Problem] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        value = node.value
        if not value.endswith(DATA_SUFFIXES):
            continue
        if "://" in value or is_placeholder(value):
            continue
        # A bare filename is usually joined to a directory elsewhere (root_dir /
        # "data" / name), so only judge values that look like a path.
        if "/" not in value:
            continue
        candidates = [
            REPO_ROOT / value,
            path.parent / value,
            EXAMPLES_DIR / value,
        ]
        if any(candidate.exists() for candidate in candidates):
            continue
        problems.append(
            Problem(
                path,
                node.lineno,
                f"data file not found: {value!r}",
                "correct the path, or add the file under examples/data/",
            )
        )
    return problems


def iter_example_files() -> list[Path]:
    return sorted(p for p in EXAMPLES_DIR.rglob("*.py") if "__pycache__" not in p.parts)


def run_static_checks() -> list[FileReport]:
    reports: list[FileReport] = []
    for path in iter_example_files():
        report = FileReport(path)
        source = path.read_text()
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as exc:
            report.problems.append(
                Problem(
                    path,
                    exc.lineno or 0,
                    f"does not parse: {exc.msg}",
                    "fix the syntax error",
                )
            )
            reports.append(report)
            continue
        report.problems.extend(check_llm_usage(path, tree))
        report.problems.extend(check_data_files(path, tree))
        reports.append(report)
    return reports


# Providers --live can satisfy from an OPENAI_API_KEY alone. Anything else is a
# reason to skip, not to run.
RUNNABLE_PROVIDERS = {"openai"}


def _service_available(service: str) -> bool:
    """Whether a service an example needs is reachable right now.

    Without this a stopped Neo4j reported every local example as FAIL, which
    reads as "the examples are broken" when the truth is "nothing was running".
    """
    return example_requirements.service_available(service)


def providers_used(path: Path) -> set[str]:
    """Providers an example depends on, used to decide if we can run it.

    Delegates to the shared requirement model rather than scanning for
    substrings. The substring version matched any file containing the word
    "google" and could not tell ``OpenAIEmbeddings`` from ``AzureOpenAIEmbeddings``.
    """
    return example_requirements.providers_for_source(path)


def run_live_checks(timeout: int) -> int:
    """Run the examples we have credentials and infrastructure for.

    Everything else is reported as skipped, never as passing - a green run must
    not imply coverage we do not have.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        print("--live needs OPENAI_API_KEY; nothing to run.", file=sys.stderr)
        return 1

    passed: list[Path] = []
    failed: list[tuple[Path, str]] = []
    skipped: list[tuple[Path, str]] = []

    for path in iter_example_files():
        requirements = example_requirements.analyse(path)
        if not requirements.runnable:
            skipped.append((path, "snippet, nothing to run"))
            continue
        blockers = requirements.providers - RUNNABLE_PROVIDERS
        if blockers:
            skipped.append((path, f"needs {', '.join(sorted(blockers))}"))
            continue
        unavailable = sorted(
            example_requirements.SERVICE_LABELS.get(service, service)
            for service in requirements.services
            if not _service_available(service)
        )
        if unavailable:
            skipped.append((path, f"needs {', '.join(unavailable)}"))
            continue
        try:
            result = subprocess.run(
                [sys.executable, str(path)],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            failed.append((path, f"timed out after {timeout}s"))
            continue
        if result.returncode == 0:
            passed.append(path)
        else:
            tail = (result.stderr or result.stdout).strip().splitlines()
            failed.append((path, tail[-1] if tail else "non-zero exit"))

    for path in passed:
        print(f"PASS {path.relative_to(REPO_ROOT)}")
    for path, reason in skipped:
        print(f"SKIP {path.relative_to(REPO_ROOT)} ({reason})")
    for path, reason in failed:
        print(f"FAIL {path.relative_to(REPO_ROOT)}: {reason}")

    print(
        f"\n{len(passed)} passed, {len(failed)} failed, {len(skipped)} skipped "
        "(skipped examples were NOT verified)"
    )
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live",
        action="store_true",
        help="also run the examples we have credentials for (costs money)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="per-example timeout in seconds for --live (default: 300)",
    )
    args = parser.parse_args()

    reports = run_static_checks()
    problems = [p for report in reports for p in report.problems]

    for problem in problems:
        print(problem.render())

    checked = len(reports)
    if problems:
        affected = len({p.path for p in problems})
        print(
            f"\n{len(problems)} problem(s) in {affected} of "
            f"{checked} example file(s)."
        )
        return 1

    print(f"{checked} example file(s) checked, no problems found.")

    if args.live:
        return run_live_checks(args.timeout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
