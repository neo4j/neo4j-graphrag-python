"""This example demonstrates how to use the SchemaFromTextExtractor component
to automatically extract a schema from text and save it to JSON and YAML files.

The SchemaFromTextExtractor component uses an LLM to analyze the text and identify entities,
relations, and their properties.

Note: This example requires an OpenAI API key to be set in the .env file.
"""

import asyncio
import logging
import os
from dotenv import load_dotenv

from neo4j_graphrag.experimental.components.schema import (
    SchemaFromTextExtractor,
    SchemaConfig,
)
from neo4j_graphrag.llm import OpenAILLM

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig()
logging.getLogger("neo4j_graphrag").setLevel(logging.INFO)

# Verify OpenAI API key is available
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError(
        "OPENAI_API_KEY environment variable not found. "
        "Please set it in the .env file in the root directory."
    )

# Sample text to extract schema from - it's about a company and its employees
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

# Define the file paths for saving the schema
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
)
JSON_FILE_PATH = os.path.join(OUTPUT_DIR, "extracted_schema.json")
YAML_FILE_PATH = os.path.join(OUTPUT_DIR, "extracted_schema.yaml")


async def extract_and_save_schema() -> None:
    """Extract schema from text and save it to JSON and YAML files."""

    # Define LLM parameters
    llm_model_params = {
        "max_tokens": 2000,
        "response_format": {"type": "json_object"},
        "temperature": 0,  # Lower temperature for more consistent output
    }

    # Create the LLM instance
    llm = OpenAILLM(
        model_name="gpt-4o",
        model_params=llm_model_params,
    )

    try:
        # Create a SchemaFromTextExtractor component with the default template
        schema_extractor = SchemaFromTextExtractor(llm=llm)

        print("Extracting schema from text...")
        # Extract schema from text
        inferred_schema = await schema_extractor.run(text=TEXT)

        # Ensure the output directory exists
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        print(f"Saving schema to JSON file: {JSON_FILE_PATH}")
        # Save the schema to JSON file
        inferred_schema.store_as_json(JSON_FILE_PATH)

        print(f"Saving schema to YAML file: {YAML_FILE_PATH}")
        # Save the schema to YAML file
        inferred_schema.store_as_yaml(YAML_FILE_PATH)

        print("\nExtracted Schema Summary:")
        print(f"Entities: {list(inferred_schema.entities.keys())}")
        print(
            f"Relations: {list(inferred_schema.relations.keys() if inferred_schema.relations else [])}"
        )

        if inferred_schema.potential_schema:
            print("\nPotential Schema:")
            for entity1, relation, entity2 in inferred_schema.potential_schema:
                print(f"  {entity1} --[{relation}]--> {entity2}")

    finally:
        # Close the LLM client
        await llm.async_client.close()


async def main() -> None:
    """Run the example."""

    # extract schema and save to files
    await extract_and_save_schema()

    print("\nSchema files have been saved to:")
    print(f"  - JSON: {JSON_FILE_PATH}")
    print(f"  - YAML: {YAML_FILE_PATH}")

    # load schema from files
    print("\nLoading schemas from saved files:")
    schema_from_json = SchemaConfig.from_file(JSON_FILE_PATH)
    schema_from_yaml = SchemaConfig.from_file(YAML_FILE_PATH)
    
    print(f"Entities in JSON schema: {list(schema_from_json.entities.keys())}")
    print(f"Entities in YAML schema: {list(schema_from_yaml.entities.keys())}")


if __name__ == "__main__":
    asyncio.run(main())
