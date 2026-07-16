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
"""Backward-compatibility shim.

Components have moved from ``neo4j_graphrag.experimental.components`` to
``neo4j_graphrag.components``. To keep old imports such as::

    from neo4j_graphrag.experimental.components.types import LexicalGraphConfig

working, the meta path finder registered below intercepts any
``neo4j_graphrag.experimental.components.*`` import, strips ``experimental`` from
the path, and — *if* the module exists at its new location — redirects to it
(with a :class:`DeprecationWarning`). The *same* module object is aliased under
both names, so it is not re-executed and ``isinstance`` checks hold across paths.

Modules that do **not** exist under ``neo4j_graphrag.components`` are left alone,
so components that only ever live in the experimental namespace keep loading from
here. Use ``_EXPERIMENTAL_ONLY`` to additionally force a component to load from
the experimental namespace even when a same-named module exists outside it.
"""

import importlib
import importlib.abc
import importlib.util
import sys
import warnings
from importlib.machinery import ModuleSpec
from types import ModuleType
from typing import Optional, Sequence

_PKG = __name__  # "neo4j_graphrag.experimental.components"

#: Top-level submodule names under this package that must always be loaded from
#: the experimental namespace, never redirected — e.g. a new component that only
#: exists in experimental but happens to share a name with one outside it.
_EXPERIMENTAL_ONLY: set[str] = set()


class I(importlib.abc.Loader):
    """Aliases the module from its new location under the old (experimental) name.

    The exact same module object is shared under both names, so it is neither
    copied nor re-executed.
    """

    def __init__(self, new_name: str) -> None:
        self._new_name = new_name

    def create_module(self, spec: ModuleSpec) -> ModuleType:
        # Return the already-loaded module from its new location. The import
        # machinery then registers *this same object* under the old name
        # (sys.modules[spec.name] = module), so both names alias one object.
        return importlib.import_module(self._new_name)

    def exec_module(self, module: ModuleType) -> None:
        # No-op: the module was executed once at its new location. A normal
        # source loader would re-run the file here — that would *copy* it.
        pass


class _ComponentsRedirectFinder(importlib.abc.MetaPathFinder):
    """Redirects ``experimental.components.*`` to ``components.*`` when it moved."""

    def find_spec(
        self,
        fullname: str,
        path: Optional[Sequence[str]] = None,
        target: Optional[ModuleType] = None,
    ) -> Optional[ModuleSpec]:
        # Only intercept submodules of this package, never the package itself
        # (its real __path__ must keep pointing here so experimental-only
        # components remain importable).
        if not fullname.startswith(_PKG + "."):
            return None

        top_level = fullname[len(_PKG) + 1 :].split(".", 1)[0]
        if top_level in _EXPERIMENTAL_ONLY:
            return None

        # Strip "experimental" from the path and try to import from there.
        new_name = fullname.replace(".experimental", "", 1)
        try:
            moved = importlib.util.find_spec(new_name) is not None
        except ModuleNotFoundError:
            moved = False
        if not moved:
            # Not moved: let the normal machinery load it from experimental.
            return None

        warnings.warn(
            f"'{fullname}' has moved to '{new_name}'. Importing from the "
            "'neo4j_graphrag.experimental.components' namespace is deprecated and "
            "will be removed in version 2.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        return importlib.util.spec_from_loader(fullname, I(new_name))


# Register the finder ahead of the default path-based finders, but only once.
if not any(isinstance(f, _ComponentsRedirectFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _ComponentsRedirectFinder())
