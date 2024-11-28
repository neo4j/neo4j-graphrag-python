"""In this example, the pipeline is defined in a JSON file 'pipeline_config.json'.
According to the configuration file, some parameters will be read from the env vars
(Neo4j credentials and the OpenAI API key).
"""

import asyncio

## If env vars in a .env file, uncomment:
## (requires pip install python-dotenv)
# from dotenv import load_dotenv
# load_dotenv()
# env vars manually set for testing:
import os

from neo4j_graphrag.experimental.pipeline.config.parser import SimpleKGPipelineBuilder
from neo4j_graphrag.experimental.pipeline.pipeline import PipelineResult

os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_USER"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "password"
# os.environ["OPENAI_API_KEY"] = "sk-..."


# Text to process
TEXT = """The son of Duke Leto Atreides and the Lady Jessica, Paul is the heir of House Atreides,
an aristocratic family that rules the planet Caladan, the rainy planet, since 10191."""


async def main() -> PipelineResult:
    file_path = "examples/customize/build_graph/pipeline/simple_kg_pipeline_config.json"
    pipeline = SimpleKGPipelineBuilder.from_config_file(file_path)
    return await pipeline.run_async(text=TEXT)


if __name__ == "__main__":
    print(asyncio.run(main()))
