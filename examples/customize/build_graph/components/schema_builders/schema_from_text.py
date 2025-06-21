"""This example demonstrates how to use the SchemaFromTextExtractor component
to automatically extract a schema from text and save it to JSON and YAML files.

The SchemaFromTextExtractor component uses an LLM to analyze the text and identify entities,
relations, and their properties.

Note: This example requires an OpenAI API key to be set in the .env file.
"""

import asyncio
import logging
from pathlib import Path
from dotenv import load_dotenv

from neo4j_graphrag.experimental.components.schema import (
    SchemaFromTextExtractor,
)
from neo4j_graphrag.experimental.components.types import (
    GraphSchema,
)
from neo4j_graphrag.llm import OpenAILLM

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig()
logging.getLogger("neo4j_graphrag").setLevel(logging.INFO)

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
root_dir = Path(__file__).parents[4]
OUTPUT_DIR = str(root_dir / "data")
JSON_FILE_PATH = str(root_dir / "data" / "extracted_schema.json")
YAML_FILE_PATH = str(root_dir / "data" / "extracted_schema.yaml")


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
        Path(OUTPUT_DIR).mkdir(exist_ok=True)

        print(f"Saving schema to JSON file: {JSON_FILE_PATH}")
        # Save the schema to JSON file
        inferred_schema.save(JSON_FILE_PATH)

        print(f"Saving schema to YAML file: {YAML_FILE_PATH}")
        # Save the schema to YAML file
        inferred_schema.save(YAML_FILE_PATH)

        print("\nExtracted Schema Summary:")
        print(f"Node types: {list(inferred_schema.node_types)}")
        print(
            f"Relationship types: {list(inferred_schema.relationship_types if inferred_schema.relationship_types else [])}"
        )

        if inferred_schema.patterns:
            print("\nPatterns:")
            for entity1, relation, entity2 in inferred_schema.patterns:
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
    schema_from_json = GraphSchema.from_file(JSON_FILE_PATH)
    schema_from_yaml = GraphSchema.from_file(YAML_FILE_PATH)

    print(f"Node types in JSON schema: {list(schema_from_json.node_types)}")
    print(f"Node types in YAML schema: {list(schema_from_yaml.node_types)}")


if __name__ == "__main__":
    asyncio.run(main())
