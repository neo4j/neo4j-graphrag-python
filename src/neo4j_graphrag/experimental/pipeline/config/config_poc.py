"""Generic config for all pipelines + specific implementation for "templates"
such as the SimpleKGPipeline.
"""

import abc
import enum
import importlib
from collections import Counter
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal, Optional, Self, Union

import neo4j
from pydantic import (
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    PrivateAttr,
    RootModel,
    Tag,
    field_validator,
)
from pydantic.utils import deep_update

from neo4j_graphrag.experimental.components.pdf_loader import PdfLoader
from neo4j_graphrag.experimental.pipeline import Component, Pipeline
from neo4j_graphrag.experimental.pipeline.config.param_resolvers import PARAM_RESOLVERS
from neo4j_graphrag.experimental.pipeline.config.reader import ConfigReader
from neo4j_graphrag.experimental.pipeline.config.types import (
    ParamConfig,
    ParamToResolveConfig,
)
from neo4j_graphrag.experimental.pipeline.pipeline import PipelineResult
from neo4j_graphrag.experimental.pipeline.types import (
    ComponentDefinition,
    ConnectionDefinition,
    PipelineDefinition,
)
from neo4j_graphrag.llm import LLMInterface


class AbstractConfig(BaseModel, abc.ABC):
    """Base class for all configs.
    Provides methods to get a class from a string and resolve a parameter defined by
    a dict with a 'resolver_' key.

    Each subclass must implement a 'parse' method that returns the relevant object.
    """

    RESOLVER_KEY: ClassVar[str] = "resolver_"

    _global_data: dict[str, Any] = PrivateAttr({})

    @classmethod
    def _get_class(cls, class_path: str, optional_module: Optional[str] = None) -> type:
        """Get class from string and an optional module"""
        *modules, class_name = class_path.rsplit(".", 1)
        module_name = modules[0] if modules else optional_module
        if module_name is None:
            module_name = "."
            # logger.debug(...)
            # raise ValueError("Must specify a module to import class from")
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
        if not isinstance(param, ParamToResolveConfig):
            return param
        resolver_name = param.resolver_
        if resolver_name not in PARAM_RESOLVERS:
            raise ValueError(
                f"Resolver {resolver_name} not found in {PARAM_RESOLVERS.keys()}"
            )
        resolver_klass = PARAM_RESOLVERS[resolver_name]
        resolver = resolver_klass(self._global_data)
        return resolver.resolve(param)

    def resolve_params(self, params: dict[str, ParamConfig]) -> dict[str, Any]:
        """Resolve parameters"""
        return {
            param_name: self.resolve_param(param)
            for param_name, param in params.items()
        }

    @abc.abstractmethod
    def parse(self) -> Any:
        raise NotImplementedError()


class ObjectConfig(AbstractConfig):
    """A config class to represent an object from a class name
    and its constructor parameters.

    Since they can be included in a list, objects must have a name
    to uniquely identify them.
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
        for p in cls.REQUIRED_PARAMS:
            if p not in params_:
                raise ValueError(f"Missing parameter {p}")
        return params_

    def get_module(self) -> str:
        return self.DEFAULT_MODULE

    def get_interface(self) -> type:
        return self.INTERFACE

    def parse(self) -> Any:
        if self.class_ is None:
            raise ValueError(f"Class {self.class_} is not defined")
        klass = self._get_class(self.class_, self.get_module())
        if not issubclass(klass, self.get_interface()):
            raise ValueError(
                f"Invalid class {klass}. Expected a subclass of {self.get_interface()}"
            )
        params = self.resolve_params(self.params_)
        obj = klass(**params)
        return obj


class Neo4jDriverConfig(ObjectConfig):
    REQUIRED_PARAMS = ["uri", "user", "password"]

    @field_validator("class_", mode="before")
    @classmethod
    def validate_class(cls, class_: Any) -> str:
        if class_:
            # logger.info("Parameter class_ is not used")
            ...
        # not used
        return "not used"

    def parse(self) -> neo4j.Driver:
        params = self.resolve_params(self.params_)
        uri = params.pop("uri")
        user = params.pop("user")
        password = params.pop("password")
        driver = neo4j.GraphDatabase.driver(uri, auth=(user, password), **params)
        return driver


class Neo4jDriverType(RootModel):
    root: Union[neo4j.Driver, Neo4jDriverConfig]

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def parse(self) -> neo4j.Driver:
        if isinstance(self.root, neo4j.Driver):
            return self.root
        return self.root.parse()


class LLMConfig(ObjectConfig):
    DEFAULT_MODULE = "neo4j_graphrag.llm"
    INTERFACE = LLMInterface


class LLMType(RootModel):
    root: Union[LLMInterface, LLMConfig]

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def parse(self) -> LLMInterface:
        if isinstance(self.root, LLMInterface):
            return self.root
        return self.root.parse()


class ComponentConfig(ObjectConfig):
    run_params_: dict[str, ParamConfig] = {}

    DEFAULT_MODULE = "neo4j_graphrag.experimental.components"
    INTERFACE = Component


class ComponentType(RootModel):
    root: Union[Component, ComponentConfig]

    model_config = ConfigDict(arbitrary_types_allowed=True)


class PipelineTemplateType(str, enum.Enum):
    NONE = "none"
    SIMPLE_KG_PIPELINE = "SimpleKGPipeline"


class AbstractPipelineConfig(AbstractConfig):
    neo4j_config: dict[str, Neo4jDriverType] = {}
    llm_config: dict[str, LLMConfig] = {}
    # extra parameters values that can be used in different places of the config file
    extras: dict[str, Any] = {}

    DEFAULT_NAME: ClassVar[str] = "default"

    @field_validator("neo4j_config", mode="before")
    @classmethod
    def validate_drivers(
        cls, drivers: Union[Neo4jDriverType, dict[str, Neo4jDriverType]]
    ) -> dict[str, Neo4jDriverType]:
        if not isinstance(drivers, dict) or "params_" in drivers:
            return {cls.DEFAULT_NAME: drivers}
        return drivers

    @field_validator("llm_config", mode="before")
    @classmethod
    def validate_llms(
        cls, llms: Union[LLMType, dict[str, LLMType]]
    ) -> dict[str, LLMType]:
        if not isinstance(llms, dict) or "params_" in llms:
            return {cls.DEFAULT_NAME: llms}
        return llms

    @field_validator("llm_config", "neo4j_config", mode="after")
    @classmethod
    def validate_names(cls, lst: dict[str, Any]) -> dict[str, Any]:
        if not lst:
            return lst
        c = Counter(lst.keys())
        most_common_item = c.most_common(1)
        most_common_count = most_common_item[0][1]
        if most_common_count > 1:
            raise ValueError(f"names must be unique {most_common_item}")
        return lst

    def _resolve_component(self, config: ComponentConfig) -> Component:
        klass_path = config.class_
        if klass_path is None:
            raise ValueError(f"Class {klass_path} is not defined")
        try:
            klass = self._get_class(
                klass_path, optional_module="neo4j_graphrag.experimental.components"
            )
        except ValueError:
            raise ValueError(f"Component '{klass_path}' not found")
        component_init_params = self.resolve_params(config.params_)
        component = klass(**component_init_params)
        return component

    def _resolve_component_definition(
        self, name: str, config: ComponentType
    ) -> ComponentDefinition:
        component = config.root
        component_run_params = {}
        if not isinstance(component, Component):
            component = self._resolve_component(config.root)
            component_run_params = self.resolve_params(config.root.run_params_)
        return ComponentDefinition(
            name=name,
            component=component,
            run_params=component_run_params,
        )

    def _parse_global_data(self) -> dict[str, Any]:
        drivers = {d: config.parse() for d, config in self.neo4j_config.items()}
        llms = {llm: config.parse() for llm, config in self.llm_config.items()}
        return {
            "neo4j_config": drivers,
            "llm_config": llms,
            "extras": self.resolve_params(self.extras),
        }

    def _get_components(self) -> list[ComponentDefinition]:
        return []

    def _get_connections(self) -> list[ConnectionDefinition]:
        return []

    def parse(self) -> PipelineDefinition:
        self._global_data = self._parse_global_data()
        return PipelineDefinition(
            components=self._get_components(),
            connections=self._get_connections(),
        )


class PipelineConfig(AbstractPipelineConfig):
    component_config: dict[str, ComponentType]
    connection_config: list[ConnectionDefinition]
    template_: Literal[PipelineTemplateType.NONE] = PipelineTemplateType.NONE

    def _get_connections(self) -> list[ConnectionDefinition]:
        return self.connection_config

    def _get_components(self) -> list[ComponentDefinition]:
        return [
            self._resolve_component_definition(name, component_config)
            for name, component_config in self.component_config.items()
        ]


class TemplatePipelineConfig(AbstractPipelineConfig):
    COMPONENTS: ClassVar[list[str]] = []

    def _get_components(self) -> list[ComponentDefinition]:
        components = []
        for component_name in self.COMPONENTS:
            method = getattr(self, f"_get_{component_name}")
            component = method()
            if component is None:
                continue
            method = getattr(self, f"_get_run_params_for_{component_name}", None)
            run_params = method() if method else {}
            components.append(
                ComponentDefinition(
                    name=component_name,
                    component=component,
                    run_params=run_params,
                )
            )
        return components


class SimpleKGPipelineConfig(TemplatePipelineConfig):
    COMPONENTS: ClassVar[list[str]] = [
        "pdf_loader",
        # "splitter",
        # "chunk_embedder",
        # "extractor",
        # "writer",
        # "entity_resolver",
    ]

    template_: Literal[PipelineTemplateType.SIMPLE_KG_PIPELINE] = (
        PipelineTemplateType.SIMPLE_KG_PIPELINE
    )
    from_pdf: bool = False
    potential_schema: Optional[list[tuple[str, str, str]]] = None
    # on_error: OnError = OnError.IGNORE
    # prompt_template: Union[ERExtractionTemplate, str] = ERExtractionTemplate()
    perform_entity_resolution: bool = True
    # lexical_graph_config: Optional[LexicalGraphConfig] = None
    neo4j_database: Optional[str] = None

    pdf_loader: ComponentConfig | None = None
    kg_writer: ComponentConfig | None = None
    text_splitter: ComponentConfig | None = None
    # entities: list[SchemaEntity] = []
    # relations: list[SchemaRelation] = []

    def _get_pdf_loader(self) -> Component | None:
        if not self.from_pdf:
            return None
        if self.pdf_loader:
            return self._resolve_component(self.pdf_loader)
        return PdfLoader()


def get_discriminator_value(model: Any) -> PipelineTemplateType:
    template_ = None
    if "template_" in model:
        template_ = model["template_"]
    if hasattr(model, "template_"):
        template_ = model.template_
    return PipelineTemplateType(template_) or PipelineTemplateType.NONE


class PipelineConfigWrapper(BaseModel):
    config: Union[
        Annotated[PipelineConfig, Tag(PipelineTemplateType.NONE)],
        Annotated[SimpleKGPipelineConfig, Tag(PipelineTemplateType.SIMPLE_KG_PIPELINE)],
    ] = Field(discriminator=Discriminator(get_discriminator_value))

    def parse(self) -> PipelineDefinition:
        return self.config.parse()


class PipelineRunner:
    def __init__(self, pipeline_definition: PipelineDefinition) -> None:
        self.pipeline = Pipeline.from_definition(pipeline_definition)
        self.run_params = pipeline_definition.get_run_params()

    @classmethod
    def from_config(cls, config: AbstractPipelineConfig) -> Self:
        wrapper = PipelineConfigWrapper.model_validate({"config": config})
        return cls(wrapper.parse())

    @classmethod
    def from_config_file(cls, file_path: Union[str, Path]) -> Self:
        pipeline_definition = cls._parse(file_path)
        return cls(pipeline_definition)

    @classmethod
    def _parse(cls, file_path: Union[str, Path]) -> PipelineDefinition:
        if not isinstance(file_path, Path):
            file_path = Path(file_path)
        data = ConfigReader().read(file_path)
        wrapper = PipelineConfigWrapper.model_validate({"config": data})
        return wrapper.parse()

    async def run(self, data: dict[str, Any]) -> PipelineResult:
        run_param = deep_update(self.run_params, data)
        return await self.pipeline.run(data=run_param)


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    import asyncio

    file_path = "examples/customize/build_graph/pipeline/pipeline_config.json"
    runner = PipelineRunner.from_config_file(file_path)
    print(runner)
    # print(asyncio.run(runner.run({"splitter": {"text": "blabla"}})))

    config = SimpleKGPipelineConfig.model_validate(
        {
            "template_": PipelineTemplateType.SIMPLE_KG_PIPELINE.value,
            "neo4j_config": neo4j.GraphDatabase.driver("bolt://", auth=("", "")),
            "from_pdf": True,
        }
    )
    print(config)
    runner = PipelineRunner.from_config(config)
    print(runner.pipeline._nodes)


"""
        {
            "name_": "embedder",
            "class_": "embedder.TextChunkEmbedder",
            "params_": {
                "embedder": {
                    "resolver_": "CONFIG_ARRAY",
                    "array_": "embedder_config",
                    "name_": "openai"
                }
            }
        },

"""
