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

import neo4j
from neo4j_graphrag.experimental.components.entity_relation_extractor import (
    LLMEntityRelationExtractor,
    OnError,
)
from neo4j_graphrag.experimental.components.kg_writer import Neo4jWriter
from neo4j_graphrag.experimental.components.pdf_loader import PdfLoader
from neo4j_graphrag.experimental.components.resolver import (
    SinglePropertyExactMatchResolver,
)
from neo4j_graphrag.experimental.components.schema import (
    SchemaBuilder,
    SchemaEntity,
    SchemaProperty,
    SchemaRelation,
)
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import (
    FixedSizeSplitter,
)
from neo4j_graphrag.experimental.pipeline import Pipeline
from neo4j_graphrag.llm import LLMInterface, OpenAILLM


async def define_and_run_pipeline(
    neo4j_driver: neo4j.Driver, llm: LLMInterface
) -> None:
    """This is where we define and run the KG builder pipeline, instantiating a few
    components:
    - Text Splitter: in this example we use the fixed size text splitter
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
    pipe.add_component(PdfLoader(), "loader")
    pipe.add_component(
        FixedSizeSplitter(),
        "splitter",
    )
    pipe.add_component(SchemaBuilder(), "schema")
    pipe.add_component(
        LLMEntityRelationExtractor(
            llm=llm,
            on_error=OnError.IGNORE,
        ),
        "extractor",
    )
    pipe.add_component(Neo4jWriter(neo4j_driver), "writer")
    pipe.add_component(SinglePropertyExactMatchResolver(neo4j_driver), "resolver")
    # define the execution order of component
    # and how the output of previous components must be used
    pipe.connect("loader", "splitter", {"text": "loader.text"})
    pipe.connect("splitter", "extractor", input_config={"chunks": "splitter"})
    pipe.connect(
        "schema",
        "extractor",
        input_config={"schema": "schema", "document_info": "loader.document_info"},
    )
    pipe.connect(
        "extractor",
        "writer",
        input_config={"graph": "extractor"},
    )
    pipe.connect("writer", "resolver", {})
    # user input:
    # the initial text
    # and the list of entities and relations we are looking for
    pipe_inputs = {
        "loader": {},
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
            ],
            "relations": [
                SchemaRelation(
                    label="WORKED_FOR",
                ),
                SchemaRelation(
                    label="FRIEND",
                ),
                SchemaRelation(
                    label="ENEMY",
                ),
            ],
            "potential_schema": [
                ("Person", "WORKED_FOR", "Organization"),
                ("Person", "FRIEND", "Person"),
                ("Person", "ENEMY", "Person"),
            ],
        },
    }
    # run the pipeline for each documents
    for document in [
        "examples/pipeline/Harry Potter and the Chamber of Secrets Summary.pdf",
        "examples/pipeline/Harry Potter and the Death Hallows Summary.pdf",
    ]:
        pipe_inputs["loader"]["filepath"] = document
        await pipe.run(pipe_inputs)


async def main() -> None:
    llm = OpenAILLM(
        model_name="gpt-4o",
        model_params={
            "max_tokens": 1000,
            "response_format": {"type": "json_object"},
        },
    )
    driver = neo4j.GraphDatabase.driver(
        "bolt://localhost:7687", auth=("neo4j", "password")
    )
    await define_and_run_pipeline(driver, llm)
    driver.close()
    await llm.async_client.close()


if __name__ == "__main__":
    asyncio.run(main())
