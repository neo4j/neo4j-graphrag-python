"""This example illustrates how to get started easily with the SimpleKGPipeline
and ingest PDF into a Neo4j Knowledge Graph.

This example assumes a Neo4j db is up and running. Update the credentials below
if needed.

It's assumed Ollama is used to run a model locally.
"""

import asyncio
import ollama
from pathlib import Path

import neo4j
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.experimental.pipeline.pipeline import PipelineResult
from neo4j_graphrag.llm import LLMInterface, LLMResponse

from llama_index.embeddings.ollama import OllamaEmbedding
from neo4j_graphrag.embeddings.base import Embedder


class OllamaEmbedder(Embedder):
    def __init__(self, ollama_embedding: OllamaEmbedding) -> None:
        self.embedder = ollama_embedding

    def embed_query(self, text: str) -> list[float]:
        embedding: list[list[float]] = self.embedder.get_text_embedding_batch(
            [text], show_progress=True
        )
        return embedding[0]


ollama_embedding = OllamaEmbedding(
    model_name="qwen2",
    base_url="http://localhost:11434",
    ollama_additional_kwargs={"mirostat": 0},
)
embedder = OllamaEmbedder(ollama_embedding)

# Neo4j db infos
URI = "neo4j://localhost:7687"
AUTH = ("neo4j", "password")
DATABASE = "neo4j"


root_dir = Path(__file__).parents[4]
file_path = "examples/data/Harry Potter and the Chamber of Secrets Summary.pdf"


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
        embedder=embedder,
        entities=ENTITIES,
        relations=RELATIONS,
        potential_schema=POTENTIAL_SCHEMA,
    )
    return await kg_builder.run_async(file_path=str(file_path))


async def main() -> PipelineResult:
    class OllamaLLM(LLMInterface):
        def invoke(self, input: str) -> LLMResponse:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": input,
                    },
                ],
                options={"temperature": 0.0},
            )
            return LLMResponse(content=response["message"]["content"])

        async def ainvoke(self, input: str) -> LLMResponse:
            return self.invoke(input)  # TODO: implement async with ollama.AsyncClient

    llm = OllamaLLM("llama3.1")
    with neo4j.GraphDatabase.driver(URI, auth=AUTH, database=DATABASE) as driver:
        res = await define_and_run_pipeline(driver, llm)

    return res


if __name__ == "__main__":
    res = asyncio.run(main())
    print(res)
