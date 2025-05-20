"""This example illustrates how to get started easily with the SimpleKGPipeline
and ingest text into a Neo4j Knowledge Graph.

This example assumes a Neo4j db is up and running. Update the credentials below
if needed.

NB: when building a KG from text, no 'Document' node is created in the Knowledge Graph.
"""

import asyncio
import logging

import neo4j
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.experimental.pipeline.pipeline import PipelineResult
from neo4j_graphrag.experimental.pipeline.types.schema import (
    EntityInputType,
    RelationInputType,
)
from neo4j_graphrag.llm import LLMInterface
from neo4j_graphrag.llm import OpenAILLM

logging.basicConfig()
logging.getLogger("neo4j_graphrag").setLevel(logging.DEBUG)
# logging.getLogger("neo4j_graphrag").setLevel(logging.INFO)


# Neo4j db infos
URI = "neo4j://localhost:7687"
AUTH = ("neo4j", "password")
DATABASE = "neo4j"

# Text to process
TEXT = """The son of Duke Leto Atreides and the Lady Jessica, Paul is the heir of House Atreides,
an aristocratic family that rules the planet Caladan, the rainy planet, since 10191."""

# Instantiate Entity and Relation objects. This defines the
# entities and relations the LLM will be looking for in the text.
NODE_TYPES: list[EntityInputType] = [
    # entities can be defined with a simple label...
    "Person",
    # ... or with a dict if more details are needed,
    # such as a description:
    {"label": "House", "description": "Family the person belongs to"},
    # or a list of properties the LLM will try to attach to the entity:
    {"label": "Planet", "properties": [{"name": "weather", "type": "STRING"}]},
]
# same thing for relationships:
RELATIONSHIP_TYPES: list[RelationInputType] = [
    "PARENT_OF",
    {
        "label": "HEIR_OF",
        "description": "Used for inheritor relationship between father and sons",
    },
    {"label": "RULES", "properties": [{"name": "fromYear", "type": "INTEGER"}]},
]
PATTERNS = [
    ("Person", "PARENT_OF", "Person"),
    ("Person", "HEIR_OF", "House"),
    ("House", "RULES", "Planet"),
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
            "node_types": NODE_TYPES,
            "relationship_types": RELATIONSHIP_TYPES,
            "patterns": PATTERNS,
        },
        from_pdf=False,
        neo4j_database=DATABASE,
    )
    return await kg_builder.run_async(text=TEXT)


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
