"""This example illustrates how to get started easily with the SimpleKGPipeline
and ingest text into a Neo4j Knowledge Graph, using a JSON or YAML configuration file.

This example assumes a Neo4j db is up and running. Update the credentials below
if needed.
"""

import asyncio

from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.experimental.pipeline.pipeline import PipelineResult

# Text to process
TEXT = """The son of Duke Leto Atreides and the Lady Jessica, Paul is the heir of
House Atreides, an aristocratic family that rules the planet Caladan."""


async def main() -> PipelineResult:
    # Create an instance of the SimpleKGPipeline
    # kg_builder = SimpleKGPipeline.from_config_file("pipeline_config.yml")
    kg_builder = SimpleKGPipeline.from_config_file("pipeline_config_full.json")
    return await kg_builder.run_async(text=TEXT)


if __name__ == "__main__":
    res = asyncio.run(main())
    print(res)
