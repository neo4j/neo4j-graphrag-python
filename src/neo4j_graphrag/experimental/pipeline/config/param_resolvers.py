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
