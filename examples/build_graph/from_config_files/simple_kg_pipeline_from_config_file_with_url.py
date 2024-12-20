"""In this example, the pipeline is defined in a JSON ('simple_kg_pipeline_config.json')
or YAML ('simple_kg_pipeline_config.yaml') file.

According to the configuration file, some parameters will be read from the env vars
(Neo4j credentials and the OpenAI API key).
"""

import asyncio
import logging

## If env vars are in a .env file, uncomment:
## (requires pip install python-dotenv)
# from dotenv import load_dotenv
# load_dotenv()
# env vars manually set for testing:
import os
from pathlib import Path

from neo4j_graphrag.experimental.pipeline.config.runner import PipelineRunner
from neo4j_graphrag.experimental.pipeline.pipeline import PipelineResult

logging.basicConfig()
logging.getLogger("neo4j_graphrag").setLevel(logging.DEBUG)

os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_USER"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "password"
# os.environ["OPENAI_API_KEY"] = "sk-..."


root_dir = Path(__file__).parent
file_path = root_dir / "simple_kg_pipeline_config_url.json"


# File to process
URL = "https://raw.githubusercontent.com/neo4j/neo4j-graphrag-python/c166afc4d5abc56a5686f3da46a97ed7c07da19d/examples/data/Harry%20Potter%20and%20the%20Chamber%20of%20Secrets%20Summary.pdf"


async def main() -> PipelineResult:
    pipeline = PipelineRunner.from_config_file(file_path)
    return await pipeline.run({"file_path": URL})


if __name__ == "__main__":
    print(asyncio.run(main()))
