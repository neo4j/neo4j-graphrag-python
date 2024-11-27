import enum
from typing import Any, Literal, Optional, Sequence, Union

from pydantic import BaseModel, ConfigDict

from neo4j_graphrag.experimental.components.entity_relation_extractor import OnError
from neo4j_graphrag.experimental.components.types import LexicalGraphConfig
from neo4j_graphrag.experimental.pipeline.types import (
    EntityInputType,
    RelationInputType,
)
from neo4j_graphrag.generation.prompts import ERExtractionTemplate


class ParamResolverEnum(str, enum.Enum):
    ENV = "ENV"


class ParamToResolveConfig(BaseModel):
    pass


class ParamFromEnvConfig(ParamToResolveConfig):
    resolver_: Literal[ParamResolverEnum.ENV] = ParamResolverEnum.ENV
    var_: str


ParamConfig = Union[
    str,
    ParamFromEnvConfig,
    dict[str, Any],
]


class BasePipelineV1Config(BaseModel):
    version_: Literal["1"] = "1"


class DriverConfig(BaseModel):
    uri: ParamConfig
    user: ParamConfig
    password: ParamConfig


class ClassConfig(BaseModel):
    class_: str
    params_: dict[str, ParamConfig]


class SimpleKGPipelineConfig(BasePipelineV1Config):
    neo4j_config: DriverConfig
    llm_config: ClassConfig
    embedder_config: ClassConfig
    from_pdf: bool = False
    entities: Optional[Sequence[EntityInputType]] = None
    relations: Optional[Sequence[RelationInputType]] = None
    potential_schema: Optional[list[tuple[str, str, str]]] = None
    pdf_loader: ClassConfig | None = None
    text_splitter: ClassConfig | None = None
    kg_writer: ClassConfig | None = None
    on_error: OnError = OnError.IGNORE
    prompt_template: Union[ERExtractionTemplate, str] = ERExtractionTemplate()
    perform_entity_resolution: bool = True
    lexical_graph_config: Optional[LexicalGraphConfig] = None
    neo4j_database: Optional[str] = None

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )
