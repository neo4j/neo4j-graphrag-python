"""Build the lexical graph one chunk at a time.

`LexicalGraphBuilder.run` needs all the chunks up front. `run_for_chunk` builds
the graph for a single chunk instead, so chunks can be streamed out of a text
splitter and written (or accumulated) as they come, without holding the whole
document in memory. `combine_graphs` merges the per-chunk graphs, deduplicating
the `Document` node that each of them carries.
"""

from neo4j_graphrag.components.lexical_graph import LexicalGraphBuilder
from neo4j_graphrag.components.text_splitters.fixed_size_splitter import (
    FixedSizeSplitter,
)
from neo4j_graphrag.components.types import (
    DocumentInfo,
    LexicalGraphConfig,
    Neo4jGraph,
)


def main() -> Neo4jGraph:
    builder = LexicalGraphBuilder(
        config=LexicalGraphConfig(),  # optional
    )
    splitter = FixedSizeSplitter(chunk_size=20, chunk_overlap=5)
    document_info = DocumentInfo(path="example")

    graph = Neo4jGraph()
    # iter_chunks yields chunks lazily, and each chunk carries the uid of the
    # previous one, which is all run_for_chunk needs to create NEXT_CHUNK
    for chunk in splitter.iter_chunks("some text to split into several chunks"):
        chunk_graph = builder.run_for_chunk(chunk, document_info)
        # a KG writer could write chunk_graph here instead of accumulating it
        graph = builder.combine_graphs(graph, chunk_graph)
    return graph
