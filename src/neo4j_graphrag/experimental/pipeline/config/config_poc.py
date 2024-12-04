"""Generic config for all pipelines + specific implementation for "templates"
such as the SimpleKGPipeline.
"""

import abc
import enum
import importlib
from collections import Counter
from pathlib import Path
from typing import (
    Annotated,
    Any,
    ClassVar,
    Generic,
    Literal,
    Optional,
    TypeVar,
    Union,
)

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
from pydantic.v1.utils import deep_update
from typing_extensions import Self

from neo4j_graphrag.embeddings import Embedder
from neo4j_graphrag.experimental.components.embedder import TextChunkEmbedder
from neo4j_graphrag.experimental.components.entity_relation_extractor import (
    EntityRelationExtractor,
    LLMEntityRelationExtractor,
    OnError,
)
from neo4j_graphrag.experimental.components.kg_writer import KGWriter, Neo4jWriter
from neo4j_graphrag.experimental.components.pdf_loader import PdfLoader
from neo4j_graphrag.experimental.components.resolver import (
    EntityResolver,
    SinglePropertyExactMatchResolver,
)
from neo4j_graphrag.experimental.components.schema import (
    SchemaBuilder,
    SchemaEntity,
    SchemaRelation,
)
from neo4j_graphrag.experimental.components.text_splitters.base import TextSplitter
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import (
    FixedSizeSplitter,
)
from neo4j_graphrag.experimental.components.types import LexicalGraphConfig
from neo4j_graphrag.experimental.pipeline import Component, Pipeline
from neo4j_graphrag.experimental.pipeline.config.param_resolvers import PARAM_RESOLVERS
from neo4j_graphrag.experimental.pipeline.config.reader import ConfigReader
from neo4j_graphrag.experimental.pipeline.config.types import (
    ParamConfig,
    ParamToResolveConfig,
)
from neo4j_graphrag.experimental.pipeline.exceptions import PipelineDefinitionError
from neo4j_graphrag.experimental.pipeline.pipeline import PipelineResult
from neo4j_graphrag.experimental.pipeline.types import (
    ComponentDefinition,
    ConnectionDefinition,
    EntityInputType,
    PipelineDefinition,
    RelationInputType,
)
from neo4j_graphrag.generation.prompts import ERExtractionTemplate
from neo4j_graphrag.llm import LLMInterface


class AbstractConfig(BaseModel, abc.ABC):
    """Base class for all configs.
    Provides methods to get a class from a string and resolve a parameter defined by
    a dict with a 'resolver_' key.

    Each subclass must implement a 'parse' method that returns the relevant object.
    """

    RESOLVER_KEY: ClassVar[str] = "resolver_"

    _global_data: dict[str, Any] = PrivateAttr({})
    """Additional parameter ignored by all model_* methods."""

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
        # all ParamToResolveConfig have a resolver_ field
        resolver_name = param.resolver_
        if resolver_name not in PARAM_RESOLVERS:
            raise ValueError(
                f"Resolver {resolver_name} not found in {PARAM_RESOLVERS.keys()}"
            )
        resolver_class = PARAM_RESOLVERS[resolver_name]
        resolver = resolver_class(self._global_data)
        return resolver.resolve(param)

    def resolve_params(self, params: dict[str, ParamConfig]) -> dict[str, Any]:
        """Resolve all parameters

        Returning dict[str, Any] because parameters can be anything (str, float, list, dict...)
        """
        return {
            param_name: self.resolve_param(param)
            for param_name, param in params.items()
        }

    @abc.abstractmethod
    def parse(self, resolved_data: dict[str, Any] | None = None) -> Any:
        raise NotImplementedError()


T = TypeVar("T")
"""Generic type to help (a bit) mypy with the return type of the parse method"""


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
        return obj  # type: ignore[return-value]


class Neo4jDriverConfig(ObjectConfig[neo4j.Driver]):
    REQUIRED_PARAMS = ["uri", "user", "password"]

    @field_validator("class_", mode="before")
    @classmethod
    def validate_class(cls, class_: Any) -> str:
        """`class_` parameter is not used because we're always using the sync driver."""
        if class_:
            # logger.info("Parameter class_ is not used")
            ...
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


class PipelineType(str, enum.Enum):
    """Pipeline type:

    NONE => Pipeline
    SIMPLE_KG_PIPELINE ~> SimpleKGPipeline
    """

    NONE = "none"
    SIMPLE_KG_PIPELINE = "SimpleKGPipeline"


class AbstractPipelineConfig(AbstractConfig):
    """This class defines the fields possibly used by all pipelines: neo4j drivers, LLMs...

    neo4j_config, llm_config can be provided by user as a single item or a dict of items.
    Validators deal with type conversion so that the field in all instances is a dict of items.
    """

    neo4j_config: dict[str, Neo4jDriverType] = {}
    llm_config: dict[str, LLMType] = {}
    embedder_config: dict[str, EmbedderType] = {}
    # extra parameters values that can be used in different places of the config file
    extras: dict[str, Any] = {}

    DEFAULT_NAME: ClassVar[str] = "default"
    """Name of the default item in dict
    """

    @field_validator("neo4j_config", mode="before")
    @classmethod
    def validate_drivers(
        cls, drivers: Union[Neo4jDriverType, dict[str, Any]]
    ) -> dict[str, Any]:
        if not isinstance(drivers, dict) or "params_" in drivers:
            return {cls.DEFAULT_NAME: drivers}
        return drivers

    @field_validator("llm_config", mode="before")
    @classmethod
    def validate_llms(cls, llms: Union[LLMType, dict[str, Any]]) -> dict[str, Any]:
        if not isinstance(llms, dict) or "params_" in llms:
            return {cls.DEFAULT_NAME: llms}
        return llms

    @field_validator("embedder_config", mode="before")
    @classmethod
    def validate_embedders(
        cls, embedders: Union[EmbedderType, dict[str, Any]]
    ) -> dict[str, Any]:
        if not isinstance(embedders, dict) or "params_" in embedders:
            return {cls.DEFAULT_NAME: embedders}
        return embedders

    @field_validator("llm_config", "neo4j_config", "embedder_config", mode="after")
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
        return config.parse(self._global_data)

    def _resolve_component_definition(
        self, name: str, config: ComponentType
    ) -> ComponentDefinition:
        component = config.parse(self._global_data)
        if hasattr(config, "run_params_"):
            component_run_params = self.resolve_params(config.run_params_)
        else:
            component_run_params = {}
        return ComponentDefinition(
            name=name,
            component=component,
            run_params=component_run_params,
        )

    def _parse_global_data(self) -> dict[str, Any]:
        """Global data contains data that can be referenced in other parts of the
        config. Typically, neo4j drivers and llms can be referenced in component input
        parameters (see ConfigKeyParamResolver)
        """
        drivers: dict[str, neo4j.Driver] = {
            driver_name: driver_config.parse()
            for driver_name, driver_config in self.neo4j_config.items()
        }
        llms: dict[str, LLMInterface] = {
            llm_name: llm_config.parse()
            for llm_name, llm_config in self.llm_config.items()
        }
        embedders: dict[str, Embedder] = {
            embedder_name: embedder_config.parse()
            for embedder_name, embedder_config in self.embedder_config.items()
        }
        return {
            "neo4j_config": drivers,
            "llm_config": llms,
            "embedder_config": embedders,
            "extras": self.resolve_params(self.extras),
        }

    def _get_components(self) -> list[ComponentDefinition]:
        return []

    def _get_connections(self) -> list[ConnectionDefinition]:
        return []

    def parse(self, resolved_data: dict[str, Any] | None = None) -> PipelineDefinition:
        """Parse the full config and returns a PipelineDefinition object, containing instantiated
        components and a list of connections.
        """
        self._global_data = self._parse_global_data()
        return PipelineDefinition(
            components=self._get_components(),
            connections=self._get_connections(),
        )

    def get_run_params(self, user_input: dict[str, Any]) -> dict[str, Any]:
        return user_input

    def get_neo4j_driver_by_name(self, name: str) -> neo4j.Driver:
        drivers = self._global_data.get("neo4j_config", {})
        return drivers.get(name)

    def get_default_neo4j_driver(self) -> neo4j.Driver:
        return self.get_neo4j_driver_by_name(self.DEFAULT_NAME)

    def get_llm_by_name(self, name: str) -> LLMInterface:
        llms = self._global_data.get("llm_config", {})
        return llms.get(name)

    def get_default_llm(self) -> LLMInterface:
        return self.get_llm_by_name(self.DEFAULT_NAME)

    def get_embedder_by_name(self, name: str) -> Embedder:
        embedders = self._global_data.get("embedder_config", {})
        return embedders.get(name)

    def get_default_embedder(self) -> Embedder:
        return self.get_embedder_by_name(self.DEFAULT_NAME)


class PipelineConfig(AbstractPipelineConfig):
    """Configuration class for raw pipelines. Config must contain all components and connections."""

    component_config: dict[str, ComponentType]
    connection_config: list[ConnectionDefinition]
    template_: Literal[PipelineType.NONE] = PipelineType.NONE

    def _get_connections(self) -> list[ConnectionDefinition]:
        return self.connection_config

    def _get_components(self) -> list[ComponentDefinition]:
        return [
            self._resolve_component_definition(name, component_config)
            for name, component_config in self.component_config.items()
        ]


class TemplatePipelineConfig(AbstractPipelineConfig):
    """This class represent a 'template' pipeline, ie pipeline with pre-defined default
    components and fixed connections.

    Component names are defined in the COMPONENTS class var. For each of them,
    a `_get_<component_name>` method must be implemented that returns the proper
    component. Optionally, `_get_<component_name>_run_params` can be implemented to
    deal with parameters required by the component's run method.
    """

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

    def get_run_params(self, user_input: dict[str, Any]) -> dict[str, Any]:
        return {}


class SimpleKGPipelineConfig(TemplatePipelineConfig):
    COMPONENTS: ClassVar[list[str]] = [
        "pdf_loader",
        "splitter",
        "chunk_embedder",
        "schema",
        "extractor",
        "writer",
        "resolver",
    ]

    template_: Literal[PipelineType.SIMPLE_KG_PIPELINE] = (
        PipelineType.SIMPLE_KG_PIPELINE
    )

    from_pdf: bool = False
    entities: list[EntityInputType] = []
    relations: list[RelationInputType] = []
    potential_schema: Optional[list[tuple[str, str, str]]] = None
    on_error: OnError = OnError.IGNORE
    prompt_template: Union[ERExtractionTemplate, str] = ERExtractionTemplate()
    perform_entity_resolution: bool = True
    lexical_graph_config: Optional[LexicalGraphConfig] = None
    neo4j_database: Optional[str] = None

    pdf_loader: ComponentConfig | None = None
    kg_writer: ComponentConfig | None = None
    text_splitter: ComponentConfig | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_pdf_loader(self) -> PdfLoader | None:
        if not self.from_pdf:
            return None
        if self.pdf_loader:
            return self.pdf_loader.parse(self._global_data)  # type: ignore
        return PdfLoader()

    def _get_splitter(self) -> TextSplitter:
        if self.text_splitter:
            return self.text_splitter.parse(self._global_data)  # type: ignore
        return FixedSizeSplitter()

    def _get_chunk_embedder(self) -> TextChunkEmbedder:
        return TextChunkEmbedder(embedder=self.get_default_embedder())

    def _get_schema(self) -> SchemaBuilder:
        return SchemaBuilder()

    def _get_run_params_for_schema(self) -> dict[str, Any]:
        return {
            "entities": [SchemaEntity.from_text_or_dict(e) for e in self.entities],
            "relations": [SchemaRelation.from_text_or_dict(r) for r in self.relations],
            "potential_schema": self.potential_schema,
        }

    def _get_extractor(self) -> EntityRelationExtractor:
        return LLMEntityRelationExtractor(
            llm=self.get_default_llm(),
            prompt_template=self.prompt_template,
            on_error=self.on_error,
        )

    def _get_writer(self) -> KGWriter:
        if self.kg_writer:
            return self.kg_writer.parse(self._global_data)  # type: ignore
        return Neo4jWriter(driver=self.get_default_neo4j_driver())

    def _get_resolver(self) -> EntityResolver | None:
        if not self.perform_entity_resolution:
            return None
        return SinglePropertyExactMatchResolver(
            driver=self.get_default_neo4j_driver(),
        )

    def _get_connections(self) -> list[ConnectionDefinition]:
        connections = []
        if self.from_pdf:
            connections.append(
                ConnectionDefinition(
                    start="pdf_loader",
                    end="splitter",
                    input_config={"text": "pdf_loader.text"},
                )
            )
            connections.append(
                ConnectionDefinition(
                    start="schema",
                    end="extractor",
                    input_config={
                        "schema": "schema",
                        "document_info": "pdf_loader.document_info",
                    },
                )
            )
        else:
            connections.append(
                ConnectionDefinition(
                    start="schema",
                    end="extractor",
                    input_config={
                        "schema": "schema",
                    },
                )
            )
        connections.append(
            ConnectionDefinition(
                start="splitter",
                end="chunk_embedder",
                input_config={
                    "text_chunks": "splitter",
                },
            )
        )
        connections.append(
            ConnectionDefinition(
                start="chunk_embedder",
                end="extractor",
                input_config={
                    "chunks": "chunk_embedder",
                },
            )
        )
        connections.append(
            ConnectionDefinition(
                start="extractor",
                end="writer",
                input_config={
                    "graph": "extractor",
                },
            )
        )

        if self.perform_entity_resolution:
            connections.append(
                ConnectionDefinition(
                    start="writer",
                    end="resolver",
                    input_config={},
                )
            )

        return connections

    def get_run_params(self, user_input: dict[str, Any]) -> dict[str, Any]:
        run_params = {}
        if self.lexical_graph_config:
            run_params["extractor"] = {
                "lexical_graph_config": self.lexical_graph_config
            }
        text = user_input.get("text")
        file_path = user_input.get("file_path")
        if not ((text is None) ^ (file_path is None)):
            # exactly one of text or user_input must be set
            raise PipelineDefinitionError(
                "Use either 'text' (when from_pdf=False) or 'file_path' (when from_pdf=True) argument."
            )
        if self.from_pdf:
            if not file_path:
                raise PipelineDefinitionError(
                    "Expected 'file_path' argument when 'from_pdf' is True."
                )
            run_params["pdf_loader"] = {"filepath": file_path}
        else:
            if not text:
                raise PipelineDefinitionError(
                    "Expected 'text' argument when 'from_pdf' is False."
                )
            run_params["splitter"] = {"text": text}
        return run_params


def _get_discriminator_value(model: Any) -> PipelineType:
    template_ = None
    if "template_" in model:
        template_ = model["template_"]
    if hasattr(model, "template_"):
        template_ = model.template_
    return PipelineType(template_) or PipelineType.NONE


class PipelineConfigWrapper(BaseModel):
    """The pipeline config wrapper will parse the right pipeline config based on the `template_` field."""

    config: Union[
        Annotated[PipelineConfig, Tag(PipelineType.NONE)],
        Annotated[SimpleKGPipelineConfig, Tag(PipelineType.SIMPLE_KG_PIPELINE)],
    ] = Field(discriminator=Discriminator(_get_discriminator_value))

    def parse(self, resolved_data: dict[str, Any] | None = None) -> PipelineDefinition:
        return self.config.parse(resolved_data)

    def get_run_params(self, user_input: dict[str, Any]) -> dict[str, Any]:
        return self.config.get_run_params(user_input)


class PipelineRunner:
    """Pipeline runner builds a pipeline from different objects and exposes a run method to run pipeline

    Pipeline can be built from:
    - A PipelineDefinition (`__init__` method)
    - A PipelineConfig (`from_config` method)
    - A config file (`from_config_file` method)
    """

    def __init__(
        self,
        pipeline_definition: PipelineDefinition,
        config: AbstractPipelineConfig | None = None,
    ) -> None:
        self.config = config
        self.pipeline = Pipeline.from_definition(pipeline_definition)
        self.run_params = pipeline_definition.get_run_params()

    @classmethod
    def from_config(cls, config: AbstractPipelineConfig | dict[str, Any]) -> Self:
        wrapper = PipelineConfigWrapper.model_validate({"config": config})
        return cls(wrapper.parse(), config=wrapper.config)

    @classmethod
    def from_config_file(cls, file_path: Union[str, Path]) -> Self:
        if not isinstance(file_path, Path):
            file_path = Path(file_path)
        data = ConfigReader().read(file_path)
        return cls.from_config(data)

    async def run(self, data: dict[str, Any]) -> PipelineResult:
        # pipeline_conditional_run_params = self.
        if self.config:
            run_param = deep_update(self.run_params, self.config.get_run_params(data))
        else:
            run_param = deep_update(self.run_params, data)
        return await self.pipeline.run(data=run_param)
