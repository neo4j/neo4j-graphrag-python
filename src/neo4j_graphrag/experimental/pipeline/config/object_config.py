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
"""Config for all parameters that can be both provided as object instance and
config dict with 'class_' and 'params_' keys.
"""

from __future__ import annotations

import logging
from typing import (
    Any,
    ClassVar,
    Generic,
    TypeVar,
    Union,
)

import neo4j
from pydantic import (
    ConfigDict,
    Field,
    RootModel,
    field_validator,
)

from neo4j_graphrag.embeddings import Embedder
from neo4j_graphrag.experimental.pipeline import Component
from neo4j_graphrag.experimental.pipeline.config.base import AbstractConfig
from neo4j_graphrag.experimental.pipeline.config.param_resolver import (
    ParamConfig,
)
from neo4j_graphrag.llm import LLMInterface

logger = logging.getLogger(__name__)


T = TypeVar("T")
"""Generic type to help mypy with the parse method when we know the exact
expected return type (e.g. for the Neo4jDriverConfig below).
"""


class ObjectConfig(AbstractConfig, Generic[T]):
    """A config class to represent an object from a class name
    and its constructor parameters.
    """

    class_: str | None = Field(default=None, validate_default=True)
    """Path to class to be instantiated."""
    params_: dict[str, ParamConfig] = {}
    """Initialization parameters."""

    DEFAULT_MODULE: ClassVar[str] = "."
    """Default module to import the class from."""
    INTERFACE: ClassVar[type] = object
    """Constraint on the class (must be a subclass of)."""
    REQUIRED_PARAMS: ClassVar[list[str]] = []
    """List of required parameters for this object constructor."""

    @field_validator("params_")
    @classmethod
    def validate_params(cls, params_: dict[str, Any]) -> dict[str, Any]:
        """Make sure all required parameters are provided."""
        for p in cls.REQUIRED_PARAMS:
            if p not in params_:
                raise ValueError(f"Missing parameter {p}")
        return params_

    def get_module(self) -> str:
        return self.DEFAULT_MODULE

    def get_interface(self) -> type:
        return self.INTERFACE

    def parse(self, resolved_data: dict[str, Any] | None = None) -> T:
        """Import `class_`, resolve `params_` and instantiate object."""
        self._global_data = resolved_data or {}
        if self.class_ is None:
            raise ValueError("`class_` is not defined")
        klass = self._get_class(self.class_, self.get_module())
        if not issubclass(klass, self.get_interface()):
            raise ValueError(
                f"Invalid class '{klass}'. Expected a subclass of '{self.get_interface()}'"
            )
        params = self.resolve_params(self.params_)
        try:
            obj = klass(**params)
        except TypeError as e:
            raise e
        # here we still need to ignore type because _get_class returns Any
        return obj  # type: ignore[return-value]


class Neo4jDriverConfig(ObjectConfig[neo4j.Driver]):
    REQUIRED_PARAMS = ["uri", "user", "password"]

    @field_validator("class_", mode="before")
    @classmethod
    def validate_class(cls, class_: Any) -> str:
        """`class_` parameter is not used because we're always using the sync driver."""
        if class_:
            logger.info("Parameter class_ is not used for Neo4jDriverConfig")
        # not used
        return "not used"

    def parse(self, resolved_data: dict[str, Any] | None = None) -> neo4j.Driver:
        params = self.resolve_params(self.params_)
        uri = params.pop(
            "uri"
        )  # we know these params are there because of the required params validator
        user = params.pop("user")
        password = params.pop("password")
        driver = neo4j.GraphDatabase.driver(uri, auth=(user, password), **params)
        return driver


# note: using the notation with RootModel + root: <type> field
# instead of RootModel[<type>] for clarity
# but this requires the type: ignore comment below
class Neo4jDriverType(RootModel):  # type: ignore[type-arg]
    """A model to wrap neo4j.Driver and Neo4jDriverConfig objects.

    The `parse` method always returns a neo4j.Driver.
    """

    root: Union[neo4j.Driver, Neo4jDriverConfig]

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def parse(self, resolved_data: dict[str, Any] | None = None) -> neo4j.Driver:
        if isinstance(self.root, neo4j.Driver):
            return self.root
        # self.root is a Neo4jDriverConfig object
        return self.root.parse()


class LLMConfig(ObjectConfig[LLMInterface]):
    """Configuration for any LLMInterface object.

    By default, will try to import from `neo4j_graphrag.llm`.
    """

    DEFAULT_MODULE = "neo4j_graphrag.llm"
    INTERFACE = LLMInterface


class LLMType(RootModel):  # type: ignore[type-arg]
    root: Union[LLMInterface, LLMConfig]

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def parse(self, resolved_data: dict[str, Any] | None = None) -> LLMInterface:
        if isinstance(self.root, LLMInterface):
            return self.root
        return self.root.parse()


class EmbedderConfig(ObjectConfig[Embedder]):
    """Configuration for any Embedder object.

    By default, will try to import from `neo4j_graphrag.embeddings`.
    """

    DEFAULT_MODULE = "neo4j_graphrag.embeddings"
    INTERFACE = Embedder


class EmbedderType(RootModel):  # type: ignore[type-arg]
    root: Union[Embedder, EmbedderConfig]

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def parse(self, resolved_data: dict[str, Any] | None = None) -> Embedder:
        if isinstance(self.root, Embedder):
            return self.root
        return self.root.parse()


class ComponentConfig(ObjectConfig[Component]):
    """A config model for all components.

    In addition to the object config, components can have pre-defined parameters
    that will be passed to the `run` method, ie `run_params_`.
    """

    run_params_: dict[str, ParamConfig] = {}

    DEFAULT_MODULE = "neo4j_graphrag.experimental.components"
    INTERFACE = Component


class ComponentType(RootModel):  # type: ignore[type-arg]
    root: Union[Component, ComponentConfig]

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def parse(self, resolved_data: dict[str, Any] | None = None) -> Component:
        if isinstance(self.root, Component):
            return self.root
        return self.root.parse(resolved_data)
