"""Generic config for all pipelines + specific implementation for "templates"
such as the SimpleKGPipeline.
"""
import abc
import enum
import importlib
from collections import Counter
from pathlib import Path
from typing import Any, Optional, Type, ClassVar, Union, Annotated, Literal, Self

import neo4j
from pydantic import BaseModel, field_validator, Tag, Discriminator, Field

from neo4j_graphrag.experimental.pipeline import Component, Pipeline
from neo4j_graphrag.experimental.pipeline.config.reader import ConfigReader
from neo4j_graphrag.experimental.pipeline.config.template_parser.param_resolvers import PARAM_RESOLVERS
from neo4j_graphrag.experimental.pipeline.types import ComponentDefinition, \
    PipelineDefinition
from neo4j_graphrag.llm import LLMInterface


class AbstractConfig(BaseModel, abc.ABC):
    """Base class for all configs.
    Provides methods to get a class from a string and resolve a parameter defined by
    a dict with a 'resolver_' key.

    Each subclass must implement a 'parse' method that returns the relevant object.
    """
    RESOLVER_KEY: ClassVar[str] = "resolver_"

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
        return klass

    @classmethod
    def resolve_params(cls, param: dict[str, Any], global_data: dict[str, Any]) -> Any:
        """Resolve parameter"""
        # if isinstance(param, list):
        #     return [cls.resolve_params(p, global_data) for p in param]
        if not isinstance(param, dict):
            return param
        if not cls.RESOLVER_KEY in param:
            return {key: cls.resolve_params(param[key], global_data) for key in param}
        resolver_name = param.pop(cls.RESOLVER_KEY)
        resolver_klass = PARAM_RESOLVERS[resolver_name]
        resolver = resolver_klass(global_data)
        return resolver.resolve(**param)

    @abc.abstractmethod
    def parse(self) -> Any:
        raise NotImplementedError()


class ObjectConfig(AbstractConfig):
    """A config class to represent an object from a class name
    and its constructor parameters.

    Since they can be included in a list, objects must have a name
    to uniquely identify them.
    """
    class_: str | None = None
    """Path to class to be instantiated."""
    name_: str = "default"
    """Object name in an array of objects."""
    params_: dict[str, Any] = {}
    """Initialization parameters."""

    DEFAULT_MODULE: ClassVar[str] = "."
    """Default module to import the class from."""
    INTERFACE: ClassVar[Type] = object
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

    def get_interface(self) -> Type:
        return self.INTERFACE

    def parse(self) -> Any:
        if self.class_ is None:
            raise ValueError(f"Class {self.class_} is not defined")
        klass = self._get_class(self.class_, self.get_module())
        if not issubclass(klass, self.get_interface()):
            raise ValueError(
                f"Invalid class {klass}. Expected a subclass of {self.get_interface()}")
        params = self.resolve_params(self.params_, {})
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
        params = self.resolve_params(self.params_, {})
        uri = params.pop("uri")
        user = params.pop("user")
        password = params.pop("password")
        driver = neo4j.GraphDatabase.driver(
            uri,
            auth=(user, password),
            **params
        )
        return driver


class LLMConfig(ObjectConfig):
    DEFAULT_MODULE = "neo4j_graphrag.llm"
    INTERFACE = LLMInterface


class ComponentConfig(ObjectConfig):
    run_params_: dict[str, Any] = {}

    DEFAULT_MODULE = "neo4j_graphrag.experimental.components"
    INTERFACE = Component


class PipelineTemplateType(str, enum.Enum):
    NONE = "none"
    SIMPLE_KG_PIPELINE = "SimpleKGPipeline"


class AbstractPipelineConfig(AbstractConfig, abc.ABC):
    neo4j_config: list[Neo4jDriverConfig]
    llm_config: list[LLMConfig]
    extras: dict[str, Any] = {}

    @field_validator("neo4j_config", mode="before")
    @classmethod
    def validate_drivers(cls, drivers: Union[Any, list[Any]]) -> list[Any]:
        if not isinstance(drivers, list):
            drivers = [drivers]
        return drivers

    @field_validator("llm_config", mode="before")
    @classmethod
    def validate_llms(cls, llms: Union[Any, list[Any]]) -> list[Any]:
        if not isinstance(llms, list):
            llms = [llms]
        return llms

    @field_validator("llm_config", "neo4j_config", mode="after")
    @classmethod
    def validate_names(cls, lst: list[Any]) -> list[Any]:
        if not lst:
            return lst
        c = Counter([item.name_ for item in lst])
        most_common_item = c.most_common(1)
        most_common_count = most_common_item[0][1]
        if most_common_count > 1:
            raise ValueError(f"names must be unique {most_common_item}")
        return lst

    def _resolve_component(self, config: ComponentConfig, global_data: dict[str, Any]) -> ComponentDefinition:
        klass_path = config.class_
        try:
            klass = self._get_class(
                klass_path, optional_module="neo4j_graphrag.experimental.components"
            )
        except ValueError:
            raise ValueError(f"Component '{klass_path}' not found")
        component_init_params = self.resolve_params(config.params_, global_data)
        component = klass(**component_init_params)
        component_run_params = self.resolve_params(config.run_params_, global_data)
        return ComponentDefinition(
            name=config.name_,
            component=component,
            run_params=component_run_params,
        )

    def _parse_global_data(self) -> dict[str, Any]:
        drivers = {
            d.name_: d.parse() for d in self.neo4j_config
        }
        llms = {
            llm.name_: llm.parse() for llm in self.llm_config
        }
        return {
            "neo4j_config": drivers,
            "llm_config": llms,
            "extras": self.resolve_params(self.extras, {}),
        }

    @abc.abstractmethod
    def _get_components(self, global_data: dict[str, Any]) -> list[ComponentDefinition]:
        ...

    def parse(self) -> PipelineDefinition:
        global_data = self._parse_global_data()
        return PipelineDefinition(
            components=self._get_components(global_data),
            connections=[]
        )


class PipelineConfig(AbstractPipelineConfig):
    component_config: list[ComponentConfig]
    template_: Literal[PipelineTemplateType.NONE] = PipelineTemplateType.NONE

    def _get_components(self, global_data: dict[str, Any]) -> list[ComponentDefinition]:
        return [
            self._resolve_component(component_config, global_data)
            for component_config in self.component_config
        ]


class TemplatePipelineConfig(AbstractPipelineConfig):
    COMPONENTS: ClassVar[list[str]] = []

    def _get_components(self, global_data: dict[str, Any]) -> list[ComponentDefinition]:
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
        "splitter",
        "chunk_embedder",
        "extractor",
        "writer",
        "entity_resolver",
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

    def get_pdf_loader(self) -> Component | None:
        if not self.from_pdf:
            return None


def get_discriminator_value(model: dict[str, Any]) -> str:
    if "template_" in model:
        return model["template_"] or PipelineTemplateType.SIMPLE_KG_PIPELINE.value
    return PipelineTemplateType.NONE.value


class PipelineConfigWrapper(BaseModel):
    config: Union[
        Annotated[PipelineConfig, Tag(PipelineTemplateType.NONE.value)],
        Annotated[SimpleKGPipelineConfig, Tag(PipelineTemplateType.SIMPLE_KG_PIPELINE.value)],
    ] = Field(discriminator=Discriminator(get_discriminator_value))

    def parse(self) -> PipelineDefinition:
        return self.config.parse()


class PipelineRunner:
    def __init__(self, pipeline_definition: PipelineDefinition) -> None:
        self.pipeline = Pipeline.from_definition(pipeline_definition)
        self.run_params = pipeline_definition.get_run_params()

    @classmethod
    def from_config_file(cls, file_path: Union[str, Path]) -> Self:
        pipeline_definition = cls._parse(file_path)
        return cls(pipeline_definition)

    @classmethod
    def _parse(cls, file_path: Union[str, Path]) -> PipelineDefinition:
        if not isinstance(file_path, Path):
            file_path = Path(file_path)
        data = ConfigReader().read(file_path)
        wrapper = PipelineConfigWrapper.model_validate(config=data)
        return wrapper.parse()

    def run(self, **data):
        print(self.pipeline._nodes)
        print(self.pipeline._edges)
        return self.run_params


if __name__ == "__main__":

    from dotenv import load_dotenv
    load_dotenv()

    file_path = "examples/customize/build_graph/pipeline/pipeline_config.json"
    runner = PipelineRunner.from_config_file(file_path)
    print(runner.run())

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
