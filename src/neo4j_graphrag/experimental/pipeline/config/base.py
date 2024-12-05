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
"""Abstract class for all pipeline configs."""

from __future__ import annotations

import importlib
import logging
from typing import Any, Optional

from pydantic import BaseModel, PrivateAttr

from neo4j_graphrag.experimental.pipeline.config.param_resolver import (
    ParamConfig,
    ParamToResolveConfig,
)

logger = logging.getLogger(__name__)


class AbstractConfig(BaseModel):
    """Base class for all configs.
    Provides methods to get a class from a string and resolve a parameter defined by
    a dict with a 'resolver_' key.

    Each subclass must implement a 'parse' method that returns the relevant object.
    """

    _global_data: dict[str, Any] = PrivateAttr({})
    """Additional parameter ignored by all Pydantic model_* methods."""

    @classmethod
    def _get_class(cls, class_path: str, optional_module: Optional[str] = None) -> type:
        """Get class from string and an optional module

        Will first try to import the class from `class_path` alone. If it results in an ImportError,
        will try to import from `f'{optional_module}.{class_path}'`

        Args:
            class_path (str): Class path with format 'my_module.MyClass'.
            optional_module (Optional[str]): Optional module path. Used to provide a default path for some known objects and simplify the notation.

        Raises:
            ValueError: if the class can't be imported, even using the optional module.
        """
        *modules, class_name = class_path.rsplit(".", 1)
        module_name = modules[0] if modules else optional_module
        if module_name is None:
            raise ValueError("Must specify a module to import class from")
        try:
            module = importlib.import_module(module_name)
            klass = getattr(module, class_name)
        except (ImportError, AttributeError):
            if optional_module and module_name != optional_module:
                full_klass_path = optional_module + "." + class_path
                return cls._get_class(full_klass_path)
            raise ValueError(f"Could not find {class_name} in {module_name}")
        return klass  # type: ignore[no-any-return]

    def resolve_param(self, param: ParamConfig) -> Any:
        """Finds the parameter value from its definition."""
        if not isinstance(param, ParamToResolveConfig):
            # some parameters do not have to be resolved, real
            # values are already provided
            return param
        return param.resolve(self._global_data)

    def resolve_params(self, params: dict[str, ParamConfig]) -> dict[str, Any]:
        """Resolve all parameters

        Returning dict[str, Any] because parameters can be anything (str, float, list, dict...)
        """
        return {
            param_name: self.resolve_param(param)
            for param_name, param in params.items()
        }

    def parse(self, resolved_data: dict[str, Any] | None = None) -> Any:
        raise NotImplementedError()
