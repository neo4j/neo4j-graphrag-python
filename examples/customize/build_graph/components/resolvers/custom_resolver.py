"""The base EntityResolver class does not enforce
a specific signature for the run method, which makes it very flexible.
"""

from typing import Any, Optional, Union

import neo4j
from neo4j_graphrag.experimental.components.resolver import EntityResolver
from neo4j_graphrag.experimental.components.types import ResolutionStats


class MyEntityResolver(EntityResolver):
    def __init__(
        self,
        driver: Union[neo4j.Driver, neo4j.AsyncDriver],
        filter_query: Optional[str] = None,
    ) -> None:
        super().__init__(driver, filter_query)

    async def run(self, *args: Any, **kwargs: Any) -> ResolutionStats:
        # logic here
        return ResolutionStats(
            number_of_nodes_to_resolve=0,
            number_of_created_nodes=0,
        )
