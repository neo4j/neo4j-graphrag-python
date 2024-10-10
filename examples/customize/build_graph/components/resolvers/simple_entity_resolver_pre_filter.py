"""The SinglePropertyExactMatchResolver merge nodes with same label
and exact same property value (by default using the "name" property).

If some nodes need to be excluded from the resolution, for instance nodes
created from a previous run, a "WHERE" query can be added. The only variable
in the query scope is "entity".

WARNING: this process is destructive, initial nodes are deleted and replaced
by the resolved ones, but all relationships are kept.
See apoc.refactor.mergeNodes documentation for more details.
"""

import neo4j
from neo4j_graphrag.experimental.components.resolver import (
    SinglePropertyExactMatchResolver,
)
from neo4j_graphrag.experimental.components.types import ResolutionStats


async def main(driver: neo4j.Driver):
    resolver = SinglePropertyExactMatchResolver(
        driver,
        # let's filter out some entities assuming the EntityToExclude label
        # was manually added to nodes in the db
        filter_query="WHERE NOT entity:EntityToExclude",
        # another example: in some cases, we do not want to merge
        # entities whose name is John Doe because we don't know if it
        # corresponds to the same real person
        # filter_query="WHERE entity.name <> 'John Doe'",
        # optionally, change the property used for resolution (default is "name")
        # resolve_property="name",
        # and the neo4j database where data is updated
        # neo4j_database="neo4j",
    )
    res: ResolutionStats = await resolver.run()
    print(res)
