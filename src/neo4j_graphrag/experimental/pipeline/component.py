import warnings
from typing import Any


def __getattr__(name: str) -> Any:
    if name in ("Component", "DataModel"):
        warnings.warn(
            "neo4j_graphrag.experimental.pipeline.component.Component "
            "and neo4j_graphrag.experimental.pipeline.component.DataModel are deprecated"
            " and will be removed in version 2.0. "
            "Please use neo4j_graphrag.components.base.Component "
            "and neo4j_graphrag.components.base.DataModel instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        from neo4j_graphrag.components.base import Component, DataModel  # noqa

        return locals()[name]
    raise AttributeError(f"module {__name__} has no attribute {name}")
