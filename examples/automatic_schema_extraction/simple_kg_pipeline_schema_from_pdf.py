"""This example demonstrates how to use SimpleKGPipeline with automatic schema extraction
from a PDF file. When no schema is provided to SimpleKGPipeline, automatic schema extraction
is performed using the LLM.

Note: This example requires an OpenAI API key to be set in the .env file.
"""

import asyncio
import logging
import os
from dotenv import load_dotenv
import neo4j

from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.embeddings import OpenAIEmbeddings

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig()
logging.getLogger("neo4j_graphrag").setLevel(logging.INFO)

# PDF file path - replace with your own PDF file
DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
)
PDF_FILE = os.path.join(DATA_DIR, "Harry Potter and the Death Hallows Summary.pdf")


async def run_kg_pipeline_with_auto_schema() -> None:
    """Run the SimpleKGPipeline with automatic schema extraction from a PDF file."""

    # Define Neo4j connection
    uri = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password")

    # Define LLM parameters
    llm_model_params = {
        "max_tokens": 2000,
        "response_format": {"type": "json_object"},
        "temperature": 0,  # Lower temperature for more consistent output
    }

    # Initialize the Neo4j driver
    driver = neo4j.GraphDatabase.driver(uri, auth=(user, password))

    # Create the LLM instance
    llm = OpenAILLM(
        model_name="gpt-4o",
        model_params=llm_model_params,
    )

    # Create the embedder instance
    embedder = OpenAIEmbeddings()

    try:
        # Create a SimpleKGPipeline instance without providing a schema
        # This will trigger automatic schema extraction
        kg_builder = SimpleKGPipeline(
            llm=llm,
            driver=driver,
            embedder=embedder,
            from_pdf=True,
        )

        print(f"Processing PDF file: {PDF_FILE}")
        # Run the pipeline on the PDF file
        await kg_builder.run_async(file_path=PDF_FILE)

    finally:
        # Close connections
        await llm.async_client.close()
        driver.close()


async def main() -> None:
    """Run the example."""
    os.makedirs(DATA_DIR, exist_ok=True)

    # Check if the PDF file exists
    if not os.path.exists(PDF_FILE):
        print(f"Warning: PDF file not found at {PDF_FILE}")
        print("Please replace with a valid PDF file path.")
        return

    # Run the pipeline
    await run_kg_pipeline_with_auto_schema()


if __name__ == "__main__":
    asyncio.run(main())
