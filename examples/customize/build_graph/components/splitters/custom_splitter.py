from collections.abc import Iterator

from neo4j_graphrag.components.text_splitters.base import TextSplitter
from neo4j_graphrag.components.types import TextChunk


class MySplitter(TextSplitter):
    def iter_chunks(self, text: str) -> Iterator[TextChunk]:
        # your logic here
        yield TextChunk(text="", index=0)
        # optional metadata
        yield TextChunk(text="", index=1, metadata={"key": "value"})
