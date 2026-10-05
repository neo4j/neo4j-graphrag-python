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
"""Regression tests for the lazy provider barrels.

``neo4j_graphrag.llm`` and ``neo4j_graphrag.embeddings`` resolve their
per-provider classes lazily so that importing either package does not pull in
every provider SDK. These tests pin that behavior: import time stays cheap
(no heavy SDK lands in ``sys.modules``), every export still resolves, ``dir()``
stays complete, and the lazy-export maps stay in sync with ``__all__`` and the
``TYPE_CHECKING`` mirror.
"""

import subprocess
import sys

import neo4j_graphrag.embeddings as embeddings_module
import neo4j_graphrag.llm as llm_module

# The heavy SDK top-levels each provider module imports. The laziness test
# asserts none of these appear in sys.modules after importing the barrels.
HEAVY_SDK_TOP_LEVELS = (
    "anthropic",
    "boto3",
    "cohere",
    "google.genai",
    "google.cloud.aiplatform",
    "mistralai",
    "openai",
    "sentence_transformers",
    "torch",
    "vertexai",
)


def test_barrel_import_does_not_load_provider_sdks() -> None:
    """Importing the barrels must not eagerly import any provider SDK.

    Runs in a subprocess: the in-process test suite has already imported most
    of these packages, which would mask a regression back to eager imports.
    """
    source = (
        "import sys\n"
        "import neo4j_graphrag.llm\n"
        "import neo4j_graphrag.embeddings\n"
        f"loaded = {HEAVY_SDK_TOP_LEVELS!r}\n"
        "assert not [m for m in loaded if m in sys.modules], (\n"
        "    'barrel import eagerly loaded provider SDKs: '\n"
        "    f'{[m for m in loaded if m in sys.modules]}'\n"
        ")\n"
        "print('ok')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", source], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_lazy_exports_resolve() -> None:
    """Every lazily-exported name resolves to the class in its submodule."""
    for name in llm_module._LAZY_EXPORTS:
        assert getattr(llm_module, name) is not None
    for name in embeddings_module._LAZY_EXPORTS:
        assert getattr(embeddings_module, name) is not None


def test_unknown_attribute_raises() -> None:
    for module in (llm_module, embeddings_module):
        try:
            getattr(module, "NoSuchExport")
        except AttributeError as exc:
            assert "NoSuchExport" in str(exc)
        else:
            raise AssertionError(f"{module.__name__}.NoSuchExport did not raise")


def test_dir_lists_lazy_exports() -> None:
    """dir() must list lazily-exported names before first access.

    A plain __getattr__ leaves them out until something touches them, hiding
    the public API from dir(), tab completion and module inspection.
    """
    assert set(llm_module._LAZY_EXPORTS) <= set(dir(llm_module))
    assert set(embeddings_module._LAZY_EXPORTS) <= set(dir(embeddings_module))


def test_lazy_exports_match_all() -> None:
    """__all__ must not list a name that neither lazy nor eager exports provide."""
    for module, eager_exports in (
        (
            llm_module,
            (
                "BaseLLM",
                "LLMResponse",
                "LLMUsage",
                "split_http_client_kwargs",
                "validate_invoke_input",
            ),
        ),
        (embeddings_module, ("Embedder",)),
    ):
        exported = set(module._LAZY_EXPORTS) | set(eager_exports)
        assert exported == set(module.__all__), module.__name__


def test_lazy_exports_match_type_checking_mirror() -> None:
    """The TYPE_CHECKING mirror must declare exactly the lazy exports.

    The mirror exists so type checkers keep resolving the lazy names as their
    real classes instead of Any. If it drifts from _LAZY_EXPORTS, either a
    name type-checks as Any or a name is statically importable but absent at
    runtime.
    """
    import ast
    from pathlib import Path

    src_root = Path(llm_module.__file__).parent.parent
    for package_name, exports in (
        ("llm", llm_module._LAZY_EXPORTS),
        ("embeddings", embeddings_module._LAZY_EXPORTS),
    ):
        tree = ast.parse((src_root / package_name / "__init__.py").read_text())
        type_checking_imports: set[str] = set()
        for node in ast.walk(tree):
            if not (isinstance(node, ast.If) and isinstance(node.test, ast.Name)):
                continue
            if node.test.id != "TYPE_CHECKING":
                continue
            for stmt in node.body:
                if isinstance(stmt, ast.ImportFrom):
                    type_checking_imports.update(
                        alias.name for alias in stmt.names if alias.name != "*"
                    )
        assert type_checking_imports == set(exports), package_name
