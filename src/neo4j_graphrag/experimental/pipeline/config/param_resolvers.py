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

import os
from typing import Any, Optional

from .types import ParamFromEnvConfig, ParamResolverEnum, ParamToResolveConfig


class ParamResolver:
    """A base class for all parameter resolvers."""

    name: ParamResolverEnum

    def resolve(self, param: ParamToResolveConfig) -> Any:
        raise NotImplementedError


class EnvParamResolver(ParamResolver):
    """Resolve a parameter by reading its value
    in the environment variables.

    Example:

    .. code-block:: python

        import os
        os.environ["MY_ENV_VAR"] = "LOCAL"

        resolver = EnvParamResolver()
        resolver.resolve("MY_ENV_VAR")
        # Output: "LOCAL"
    """

    name = ParamResolverEnum.ENV

    def resolve(self, param: ParamFromEnvConfig) -> Optional[str]:
        return os.environ.get(param.var_)


PARAM_RESOLVERS = {
    resolver.name: resolver
    for resolver in [
        EnvParamResolver,
    ]
}
