"""The SpaCySemanticMatchResolver merges nodes with same label
and similar textual properties (by default using the "name" property) based on spaCy
embeddings and cosine similarities of embedding vectors.

If the resolution is intended to be applied only on some nodes, for instance nodes that
belong to a specific document, a "WHERE" query can be added. The only variable in the
query scope is "entity".

WARNING: this process is destructive, initial nodes are deleted and replaced
by the resolved ones, but all relationships are kept.
See apoc.refactor.mergeNodes documentation for more details.
"""

import neo4j
from neo4j_graphrag.experimental.components.resolver import (
    SpaCySemanticMatchResolver,
)
from neo4j_graphrag.experimental.components.types import ResolutionStats


async def main(driver: neo4j.Driver) -> None:
    resolver = SpaCySemanticMatchResolver(
        driver,
        # let's filter all entities that belong to a certain docId
        filter_query="WHERE (entity)-[:FROM_CHUNK]->(:Chunk)-[:FROM_DOCUMENT]->(doc:"
        "Document {id = 'docId'}",
        # optionally, change the properties used for resolution (default is "name")
        # resolve_properties=["name", "ssn"],
        # the similarity threshold (default is 0.8)
        # similarity_threshold=0.9
        # the spaCy trained model (default is "en_core_web_lg")
        # spacy_model="en_core_web_sm"
        # and the neo4j database where data is updated
        # neo4j_database="neo4j",
    )
    res: ResolutionStats = await resolver.run()
    print(res)
