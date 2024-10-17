"""The SinglePropertyExactMatchResolver merge nodes with same label
and exact same property value (by default using the "name" property).

WARNING: this process is destructive, initial nodes are deleted and replaced
by the resolved ones, but all relationships are kept.
See apoc.refactor.mergeNodes documentation for more details.
"""

import neo4j
from neo4j_graphrag.experimental.components.resolver import (
    SinglePropertyExactMatchResolver,
)
from neo4j_graphrag.experimental.components.types import ResolutionStats


async def main(driver: neo4j.Driver) -> None:
    resolver = SinglePropertyExactMatchResolver(
        driver,
        # optionally, change the property used for resolution (default is "name")
        # resolve_property="name",
        # and the neo4j database where data is updated
        # neo4j_database="neo4j",
    )
    res: ResolutionStats = await resolver.run()
    print(res)
