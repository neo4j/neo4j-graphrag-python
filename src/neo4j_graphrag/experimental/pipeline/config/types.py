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
    CONFIG_ARRAY = "CONFIG_ARRAY"
    CONFIG_KEY = "CONFIG_KEY"


class ParamToResolveConfig(BaseModel):
    pass


class ParamFromEnvConfig(ParamToResolveConfig):
    resolver_: Literal[ParamResolverEnum.ENV] = ParamResolverEnum.ENV
    var_: str


class ParamFromKeyConfig(ParamToResolveConfig):
    resolver_: Literal[ParamResolverEnum.CONFIG_KEY] = ParamResolverEnum.CONFIG_KEY
    key_: str


ParamConfig = Union[
    float,
    str,
    ParamFromEnvConfig,
    ParamFromKeyConfig,
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


class SimpleKGPipelineExposedParamConfig(BaseModel):
    from_pdf: bool = False
    potential_schema: Optional[list[tuple[str, str, str]]] = None
    on_error: OnError = OnError.IGNORE
    prompt_template: Union[ERExtractionTemplate, str] = ERExtractionTemplate()
    perform_entity_resolution: bool = True
    lexical_graph_config: Optional[LexicalGraphConfig] = None
    neo4j_database: Optional[str] = None

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )


class SimpleKGPipelineConfig(BasePipelineV1Config, SimpleKGPipelineExposedParamConfig):
    neo4j_config: DriverConfig
    llm_config: ClassConfig
    embedder_config: ClassConfig
    pdf_loader: ClassConfig | None = None
    text_splitter: ClassConfig | None = None
    kg_writer: ClassConfig | None = None
    entities: Optional[Sequence[EntityInputType]] = None
    relations: Optional[Sequence[RelationInputType]] = None
