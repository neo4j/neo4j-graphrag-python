from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import (
    FixedSizeSplitter,
)
from neo4j_graphrag.experimental.components.types import TextChunks


async def main() -> TextChunks:
    splitter = FixedSizeSplitter(
        # optionally, configure chunk_size, chunk_overlap, and approximate flag
        # chunk_size=4000,
        # chunk_overlap=200,
        # approximate = False
    )
    chunks = await splitter.run(text="text to split")
    return chunks
