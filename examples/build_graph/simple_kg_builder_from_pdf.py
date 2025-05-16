"""This example illustrates how to get started easily with the SimpleKGPipeline
and ingest PDF into a Neo4j Knowledge Graph.

This example assumes a Neo4j db is up and running. Update the credentials below
if needed.

OPENAI_API_KEY needs to be in the env vars.
"""

import asyncio
from pathlib import Path

import neo4j
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.experimental.pipeline.pipeline import PipelineResult
from neo4j_graphrag.llm import LLMInterface
from neo4j_graphrag.llm import OpenAILLM

# Neo4j db infos
URI = "neo4j://localhost:7687"
AUTH = ("neo4j", "password")
DATABASE = "neo4j"


root_dir = Path(__file__).parents[1]
file_path = root_dir / "data" / "Harry Potter and the Chamber of Secrets Summary.pdf"


# Instantiate Entity and Relation objects. This defines the
# entities and relations the LLM will be looking for in the text.
ENTITIES = ["Person", "Organization", "Location"]
RELATIONS = ["SITUATED_AT", "INTERACTS", "LED_BY"]
POTENTIAL_SCHEMA = [
    ("Person", "SITUATED_AT", "Location"),
    ("Person", "INTERACTS", "Person"),
    ("Organization", "LED_BY", "Person"),
]


async def define_and_run_pipeline(
    neo4j_driver: neo4j.Driver,
    llm: LLMInterface,
) -> PipelineResult:
    # Create an instance of the SimpleKGPipeline
    kg_builder = SimpleKGPipeline(
        llm=llm,
        driver=neo4j_driver,
        embedder=OpenAIEmbeddings(),
        schema={
            "entities": ENTITIES,
            "relations": RELATIONS,
            "potential_schema": POTENTIAL_SCHEMA,
        },
        neo4j_database=DATABASE,
    )
    return await kg_builder.run_async(file_path=str(file_path))


async def main() -> PipelineResult:
    llm = OpenAILLM(
        model_name="gpt-4o",
        model_params={
            "max_tokens": 2000,
            "response_format": {"type": "json_object"},
        },
    )
    with neo4j.GraphDatabase.driver(URI, auth=AUTH) as driver:
        res = await define_and_run_pipeline(driver, llm)
    await llm.async_client.close()
    return res


if __name__ == "__main__":
    res = asyncio.run(main())
    print(res)
