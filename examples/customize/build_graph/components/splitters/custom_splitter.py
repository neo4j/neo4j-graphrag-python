from neo4j_graphrag.components.text_splitters.base import TextSplitter
from neo4j_graphrag.components.types import (
    TextChunk,
    TextChunks,
)


class MySplitter(TextSplitter):
    async def run(self, text: str) -> TextChunks:
        # your logic here
        return TextChunks(
            chunks=[
                TextChunk(text="", index=0),
                # optional metadata
                TextChunk(text="", index=1, metadata={"key": "value"}),
            ]
        )
