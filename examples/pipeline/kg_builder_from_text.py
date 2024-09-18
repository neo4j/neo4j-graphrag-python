#  Copyright (c) "Neo4j"
#  Neo4j Sweden AB [https://neo4j.com]
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#      https://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
from __future__ import annotations

import asyncio
import logging.config

import neo4j
from langchain_text_splitters import CharacterTextSplitter
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.experimental.components.embedder import TextChunkEmbedder
from neo4j_graphrag.experimental.components.entity_relation_extractor import (
    LLMEntityRelationExtractor,
    OnError,
)
from neo4j_graphrag.experimental.components.kg_writer import Neo4jWriter
from neo4j_graphrag.experimental.components.schema import (
    SchemaBuilder,
    SchemaEntity,
    SchemaProperty,
    SchemaRelation,
)
from neo4j_graphrag.experimental.components.text_splitters.langchain import (
    LangChainTextSplitterAdapter,
)
from neo4j_graphrag.experimental.pipeline import Pipeline
from neo4j_graphrag.experimental.pipeline.pipeline import PipelineResult
from neo4j_graphrag.llm.openai import OpenAILLM

# set log level to DEBUG for all neo4j_graphrag.* loggers
logging.config.dictConfig(
    {
        "version": 1,
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
            }
        },
        "loggers": {
            "root": {
                "handlers": ["console"],
            },
            "neo4j_graphrag": {
                "level": "DEBUG",
            },
        },
    }
)


async def main(neo4j_driver: neo4j.Driver) -> PipelineResult:
    """This is where we define and run the KG builder pipeline, instantiating a few
    components:
    - Text Splitter: in this example we use a text splitter from the LangChain package
    - Schema Builder: this component takes a list of entities, relationships and
        possible triplets as inputs, validate them and return a schema ready to use
        for the rest of the pipeline
    - LLM Entity Relation Extractor is an LLM-based entity and relation extractor:
        based on the provided schema, the LLM will do its best to identity these
        entities and their relations within the provided text
    - KG writer: once entities and relations are extracted, they can be writen
        to a Neo4j database
    """
    pipe = Pipeline()
    # define the components
    pipe.add_component(
        LangChainTextSplitterAdapter(
            # chunk_size=50 for the sake of this demo
            CharacterTextSplitter(chunk_size=50, chunk_overlap=10, separator=".")
        ),
        "splitter",
    )
    pipe.add_component(TextChunkEmbedder(embedder=OpenAIEmbeddings()), "chunk_embedder")
    pipe.add_component(SchemaBuilder(), "schema")
    pipe.add_component(
        LLMEntityRelationExtractor(
            llm=OpenAILLM(
                model_name="gpt-4o",
                model_params={
                    "max_tokens": 1000,
                    "response_format": {"type": "json_object"},
                },
            ),
            on_error=OnError.RAISE,
        ),
        "extractor",
    )
    pipe.add_component(Neo4jWriter(neo4j_driver), "writer")
    # define the execution order of component
    # and how the output of previous components must be used
    pipe.connect("splitter", "chunk_embedder", input_config={"text_chunks": "splitter"})
    pipe.connect("schema", "extractor", input_config={"schema": "schema"})
    pipe.connect(
        "chunk_embedder", "extractor", input_config={"chunks": "chunk_embedder"}
    )
    pipe.connect(
        "extractor",
        "writer",
        input_config={"graph": "extractor"},
    )
    # user input:
    # the initial text
    # and the list of entities and relations we are looking for
    pipe_inputs = {
        "splitter": {
            "text": """Albert Einstein was a German physicist born in 1879 who
            wrote many groundbreaking papers especially about general relativity
            and quantum mechanics. He worked for many different institutions, including
            the University of Bern in Switzerland and the University of Oxford."""
        },
        "schema": {
            "entities": [
                SchemaEntity(
                    label="Person",
                    properties=[
                        SchemaProperty(name="name", type="STRING"),
                        SchemaProperty(name="place_of_birth", type="STRING"),
                        SchemaProperty(name="date_of_birth", type="DATE"),
                    ],
                ),
                SchemaEntity(
                    label="Organization",
                    properties=[
                        SchemaProperty(name="name", type="STRING"),
                        SchemaProperty(name="country", type="STRING"),
                    ],
                ),
                SchemaEntity(
                    label="Field",
                    properties=[
                        SchemaProperty(name="name", type="STRING"),
                    ],
                ),
            ],
            "relations": [
                SchemaRelation(
                    label="WORKED_ON",
                ),
                SchemaRelation(
                    label="WORKED_FOR",
                ),
            ],
            "potential_schema": [
                ("Person", "WORKED_ON", "Field"),
                ("Person", "WORKED_FOR", "Organization"),
            ],
        },
        "extractor": {
            "document_info": {
                "path": "my text",
            }
        },
    }
    # run the pipeline
    return await pipe.run(pipe_inputs)


if __name__ == "__main__":
    with neo4j.GraphDatabase.driver(
        "bolt://localhost:7687", auth=("neo4j", "password")
    ) as driver:
        print(asyncio.run(main(driver)))
