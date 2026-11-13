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
import sys
from typing import TYPE_CHECKING, Any

# As in ``neo4j_graphrag.llm``, provider classes are imported the first time that
# they are used, so importing this package doesn't import an HTTP client.
from neo4j_graphrag.utils.lazy_import import lazy_dir, lazy_getattr

from .base import DecisionModelInterface, DecisionState
from .types import DecisionResponse, DecisionUsage

_LAZY_EXPORTS: dict[str, str] = {
    "OpenRouterDecisionModel": ".openrouter",
}

if TYPE_CHECKING:
    # Lets type checkers see the provider classes. It has no effect at runtime.
    from .openrouter import OpenRouterDecisionModel

__all__ = [
    "DecisionModelInterface",
    "DecisionResponse",
    "DecisionState",
    "DecisionUsage",
    "OpenRouterDecisionModel",
]


def __getattr__(name: str) -> Any:
    """Import a provider class the first time that it is used."""
    return lazy_getattr(name, _LAZY_EXPORTS, sys.modules[__name__])


def __dir__() -> list[str]:
    """List the provider classes in ``dir(neo4j_graphrag.decision)`` too."""
    return lazy_dir(_LAZY_EXPORTS, sys.modules[__name__])
