from collections.abc import Iterator

from neo4j_graphrag.components.text_splitters.base import TextSplitter
from neo4j_graphrag.components.types import TextChunk


class MySplitter(TextSplitter):
    def iter_chunks(self, text: str) -> Iterator[TextChunk]:
        # your logic here
        chunk = TextChunk(text="", index=0)
        yield chunk
        # optional metadata, and prev_chunk_id to link chunks together:
        # setting it lets the lexical graph builder create the NEXT_CHUNK
        # relationship from a single chunk (see LexicalGraphBuilder.run_for_chunk).
        # LexicalGraphBuilder.run infers it from the chunk order when unset.
        yield TextChunk(
            text="",
            index=1,
            prev_chunk_id=chunk.chunk_id,
            metadata={"key": "value"},
        )
