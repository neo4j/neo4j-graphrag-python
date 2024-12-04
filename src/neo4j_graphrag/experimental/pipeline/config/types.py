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
from typing import Any, Literal, Union

from pydantic import BaseModel


class ParamResolverEnum(str, enum.Enum):
    ENV = "ENV"
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
