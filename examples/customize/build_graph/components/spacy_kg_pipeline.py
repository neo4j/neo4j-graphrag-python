# Install: pip install 'neo4j-graphrag[nlp]'
# Then:    python -m spacy download en_core_web_sm
"""End-to-end SpaCy knowledge-graph pipeline example.

Demonstrates how to wire four pipeline components together without an LLM:

    PdfLoader
        → HierarchicalTextSplitter
        → SpacyEntityRelationExtractor
        → Neo4jWriter

This example assumes a Neo4j instance is running locally. Update the
credentials below if needed.

The pipeline extracts (subject, predicate, object) triples from the document
using SpaCy's dependency parser and named-entity recogniser and writes the
resulting nodes and relationships to Neo4j.
"""

import asyncio
from pathlib import Path

import neo4j

from neo4j_graphrag.experimental.components.data_loader import PdfLoader
from neo4j_graphrag.experimental.components.kg_writer import Neo4jWriter
from neo4j_graphrag.experimental.components.spacy_entity_relation_extractor import (
    SpacyEntityRelationExtractor,
)
from neo4j_graphrag.experimental.components.text_splitters.hierarchical_splitter import (
    HierarchicalTextSplitter,
)

# ---------------------------------------------------------------------------
# Configuration — update to match your environment
# ---------------------------------------------------------------------------

URI = "neo4j://localhost:7687"
AUTH = ("neo4j", "password")
DATABASE = "neo4j"

root_dir = Path(__file__).parents[4]
FILE_PATH = root_dir / "data" / "Harry Potter and the Chamber of Secrets Summary.pdf"


# ---------------------------------------------------------------------------
# Pipeline definition
# ---------------------------------------------------------------------------


async def run_pipeline(driver: neo4j.Driver) -> None:
    """Load a PDF, split it, extract entities/relations, and write to Neo4j."""

    # 1. Load the PDF into a single Document string.
    loader = PdfLoader()
    document = await loader.run(filepath=FILE_PATH)

    # 2. Split the document text into overlapping chunks, respecting section
    #    boundaries detected from Markdown-style headers.
    splitter = HierarchicalTextSplitter(
        # Maximum number of characters per chunk.
        max_chunk_size=2048,
        # Overlap between consecutive chunks to preserve context at boundaries.
        chunk_overlap=200,
        # Header detection strategy; 'markdown' looks for ATX-style # headers.
        header_strategy="markdown",
        # drop_verbless_sentences=True (default) filters noisy sentence fragments.
    )
    chunks = await splitter.run(text=document.text)

    # 3. Extract entities and relationships from each chunk using SpaCy.
    #    No LLM is required — extraction is performed entirely locally.
    extractor = SpacyEntityRelationExtractor(
        model="en_core_web_sm",
        # Use both dependency-based and proximity-based triple extraction.
        use_linear_extractor=True,
        # Add Chunk and Document nodes to the graph for provenance tracking.
        create_lexical_graph=True,
        # on_error=OnError.IGNORE (default) skips chunks that raise exceptions.
    )
    graph = await extractor.run(
        chunks=chunks,
        document_info=document.document_info,
    )

    # 4. Write the resulting nodes and relationships to Neo4j.
    writer = Neo4jWriter(
        driver=driver,
        neo4j_database=DATABASE,
        # batch_size=1000,  # tune for throughput on large documents
    )
    result = await writer.run(graph=graph)

    print(
        f"Wrote {result.nodes_upserted} nodes and "
        f"{result.relationships_created} relationships to Neo4j."
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def main() -> None:
    with neo4j.GraphDatabase.driver(URI, auth=AUTH) as driver:
        await run_pipeline(driver)


if __name__ == "__main__":
    asyncio.run(main())
