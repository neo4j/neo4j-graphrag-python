from neo4j_graphrag.experimental.components.lexical_graph import (
    LexicalGraphBuilder,
)
from neo4j_graphrag.experimental.components.types import (
    GraphResult,
    LexicalGraphConfig,
    TextChunk,
    TextChunks,
)


async def main() -> GraphResult:
    """ """
    # optionally, define a LexicalGraphConfig object
    # shown below with default values
    config = LexicalGraphConfig(
        id_prefix="",  # used to prefix the chunk and document IDs
        chunk_node_label="Chunk",
        document_node_label="Document",
        chunk_to_document_relationship_type="PART_OF_DOCUMENT",
        next_chunk_relationship_type="NEXT_CHUNK",
        node_to_chunk_relationship_type="PART_OF_CHUNK",
        chunk_embedding_property="embeddings",
    )
    builder = LexicalGraphBuilder(
        config=config,  # optional
    )
    graph_result = await builder.run(
        text_chunks=TextChunks(chunks=[TextChunk(text="....", index=0)]),
        # document_info={"path": "example"},  # uncomment to create a "Document" node
    )
    return graph_result
