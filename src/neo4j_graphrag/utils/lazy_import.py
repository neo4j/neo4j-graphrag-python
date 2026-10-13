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
"""Helpers for packages that resolve their exports lazily.

A package that eagerly imports every provider module forces that cost on
every consumer, even ones that only use the provider-agnostic base classes.
Delegating ``__getattr__``/``__dir__`` to :func:`lazy_getattr`/
:func:`lazy_dir` backed by an explicit name-to-submodule map keeps the public
import surface unchanged while deferring the actual import to first access.
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Any


def lazy_getattr(
    name: str,
    exports: dict[str, str],
    module: ModuleType,
) -> Any:
    """Resolve a lazily-exported name on first access.

    Args:
        name: The attribute being looked up.
        exports: Map of export name to the relative submodule it lives in,
            e.g. ``{"OpenAILLM": ".openai_llm"}``. Kept explicit (rather than
            deriving a module from the name) so a symbol name and its module
            can diverge, e.g. ``BaseAnthropicLLM`` lives in ``anthropic_llm``.
        module: The package ``__getattr__`` was called for (``__name__`` is
            resolved against its ``__package__``).

    Raises:
        AttributeError: If ``name`` is not a lazily-exported name.
    """
    relative_name = exports.get(name)
    if relative_name is None:
        raise AttributeError(f"module {module.__name__!r} has no attribute {name!r}")
    value = getattr(import_module(relative_name, module.__package__), name)
    # Cache on the package: future lookups bypass __getattr__ entirely.
    setattr(module, name, value)
    return value


def lazy_dir(
    exports: dict[str, str],
    module: ModuleType,
) -> list[str]:
    """List the lazily-exported names alongside the module's own attributes.

    Without this, ``dir(package)`` omits every lazily-exported name until it
    has been accessed at least once, hiding the public API from ``dir()``,
    tab completion and anything that inspects the module.
    """
    return [*module.__dict__.keys(), *exports.keys()]
