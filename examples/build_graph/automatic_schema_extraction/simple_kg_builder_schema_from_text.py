"""This example demonstrates how to use SimpleKGPipeline with automatic schema extraction
from a text input. When no schema is provided to SimpleKGPipeline, automatic schema extraction
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
logging.getLogger("neo4j_graphrag").setLevel(logging.DEBUG)

# Sample text to build a knowledge graph from
TEXT = """
Acme Corporation was founded in 1985 by John Smith in New York City. 
The company specializes in manufacturing high-quality widgets and gadgets 
for the consumer electronics industry.

Sarah Johnson joined Acme in 2010 as a Senior Engineer and was promoted to 
Engineering Director in 2015. She oversees a team of 12 engineers working on 
next-generation products. Sarah holds a PhD in Electrical Engineering from MIT 
and has filed 5 patents during her time at Acme.

The company expanded to international markets in 2012, opening offices in London, 
Tokyo, and Berlin. Each office is managed by a regional director who reports 
directly to the CEO, Michael Brown, who took over leadership in 2008.

Acme's most successful product, the SuperWidget X1, was launched in 2018 and 
has sold over 2 million units worldwide. The product was developed by a team led 
by Robert Chen, who joined the company in 2016 after working at TechGiant for 8 years.

The company currently employs 250 people across its 4 locations and had a revenue 
of $75 million in the last fiscal year. Acme is planning to go public in 2024 
with an estimated valuation of $500 million.
"""


async def run_kg_pipeline_with_auto_schema() -> None:
    """Run the SimpleKGPipeline with automatic schema extraction from text input."""

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
            from_pdf=False,  # Using raw text input, not PDF
        )

        # Run the pipeline on the text
        await kg_builder.run_async(text=TEXT)

    finally:
        # Close connections
        await llm.async_client.close()
        driver.close()


async def main() -> None:
    """Run the example."""
    await run_kg_pipeline_with_auto_schema()


if __name__ == "__main__":
    asyncio.run(main())
