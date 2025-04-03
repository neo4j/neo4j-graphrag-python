"""The FuzzyMatchResolver merges nodes with same label
and similar textual properties (by default using the "name" property) based on RapidFuzz
for string matching.

If the resolution is intended to be applied only on some nodes, for instance nodes that
belong to a specific document, a "WHERE" query can be added. The only variable in the
query scope is "entity".

WARNING: this process is destructive, initial nodes are deleted and replaced
by the resolved ones, but all relationships are kept.
See apoc.refactor.mergeNodes documentation for more details.
"""

from neo4j_graphrag.experimental.components.resolver import (
    FuzzyMatchResolver,
)
from neo4j_graphrag.experimental.components.types import ResolutionStats

import neo4j


async def main(driver: neo4j.Driver) -> None:
    resolver = FuzzyMatchResolver(
        driver,
        # let's filter all entities that belong to a certain docId
        filter_query="WHERE (entity)-[:FROM_CHUNK]->(:Chunk)-[:FROM_DOCUMENT]->(doc:"
        "Document {id = 'docId'}",
        # optionally, change the properties used for resolution (default is "name")
        # resolve_properties=["name", "ssn"],
        # the similarity threshold (default is 0.8)
        # similarity_threshold=0.9
        # and the neo4j database where data is updated
        # neo4j_database="neo4j",
    )
    res: ResolutionStats = await resolver.run()
    print(res)
