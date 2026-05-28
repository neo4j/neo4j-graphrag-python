"""Example: using LiteParseLoader as an alternative PDF loader.

LiteParse (https://github.com/run-llama/liteparse) is a local, zero-cloud PDF
parser with optional OCR support.  Install the optional extra before running:

    pip install "neo4j-graphrag[liteparse]"

Usage::

    from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
    from neo4j_graphrag.experimental.components.data_loader import LiteParseLoader

    pipeline = SimpleKGPipeline(
        llm=...,
        driver=...,
        embedder=...,
        file_loader=LiteParseLoader(ocr_enabled=True),
        from_file=True,
    )
    await pipeline.run_async(file_path="document.pdf")
"""

# LiteParseLoader lives in the main package — import it from there.
from neo4j_graphrag.experimental.components.data_loader import LiteParseLoader

__all__ = ["LiteParseLoader"]
