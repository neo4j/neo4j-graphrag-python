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
import logging
from typing import Any, Dict, List

import neo4j
from langchain_text_splitters import CharacterTextSplitter
from neo4j_graphrag.experimental.components.entity_relation_extractor import (
    LLMEntityRelationExtractor,
    OnError,
)
from neo4j_graphrag.experimental.components.kg_writer import Neo4jWriter
from neo4j_graphrag.experimental.components.pdf_loader import PdfLoader
from neo4j_graphrag.experimental.components.schema import (
    SchemaBuilder,
    SchemaEntity,
    SchemaRelation,
)
from neo4j_graphrag.experimental.components.text_splitters.langchain import (
    LangChainTextSplitterAdapter,
)
from neo4j_graphrag.experimental.pipeline import Component, DataModel
from neo4j_graphrag.experimental.pipeline.pipeline import PipelineResult
from neo4j_graphrag.llm import OpenAILLM
from pydantic import BaseModel, validate_call

logging.basicConfig(level=logging.INFO)


class DocumentChunkModel(DataModel):
    chunks: list[str]


class DocumentChunker(Component):
    async def run(self, text: str) -> DocumentChunkModel:
        chunks = [t.strip() for t in text.split(".") if t.strip()]
        return DocumentChunkModel(chunks=chunks)


class EntityModel(BaseModel):
    label: str
    properties: dict[str, str]


class Neo4jGraph(DataModel):
    entities: list[dict[str, Any]]
    relations: list[dict[str, Any]]


class ERExtractor(Component):
    async def _process_chunk(self, chunk: str, schema: str) -> Dict[str, Any]:
        return {
            "entities": [{"label": "Person", "properties": {"name": "John Doe"}}],
            "relations": [],
        }

    async def run(self, chunks: List[str], schema: str) -> Neo4jGraph:
        tasks = [self._process_chunk(chunk, schema) for chunk in chunks]
        result = await asyncio.gather(*tasks)
        merged_result: dict[str, Any] = {"entities": [], "relations": []}
        for res in result:
            merged_result["entities"] += res["entities"]
            merged_result["relations"] += res["relations"]
        return Neo4jGraph(
            entities=merged_result["entities"], relations=merged_result["relations"]
        )


class WriterModel(DataModel):
    status: str
    entities: list[EntityModel]
    relations: list[EntityModel]


class Writer(Component):
    @validate_call
    async def run(self, graph: Neo4jGraph) -> WriterModel:
        entities = graph.entities
        relations = graph.relations
        return WriterModel(
            status="OK",
            entities=[EntityModel(**e) for e in entities],
            relations=[EntityModel(**r) for r in relations],
        )


async def main(neo4j_driver: neo4j.Driver) -> PipelineResult:
    from neo4j_graphrag.experimental.pipeline import Pipeline

    # Instantiate Entity and Relation objects
    entities = [
        SchemaEntity(label="PERSON", description="An individual human being."),
        SchemaEntity(
            label="ORGANIZATION",
            description="A structured group of people with a common purpose.",
        ),
        SchemaEntity(label="LOCATION", description="A location or place."),
        SchemaEntity(
            label="HORCRUX",
            description="A magical item in the Harry Potter universe.",
        ),
    ]
    relations = [
        SchemaRelation(
            label="SITUATED_AT", description="Indicates the location of a person."
        ),
        SchemaRelation(
            label="LED_BY",
            description="Indicates the leader of an organization.",
        ),
        SchemaRelation(
            label="OWNS",
            description="Indicates the ownership of an item such as a Horcrux.",
        ),
        SchemaRelation(
            label="INTERACTS", description="The interaction between two people."
        ),
    ]
    potential_schema = [
        ("PERSON", "SITUATED_AT", "LOCATION"),
        ("PERSON", "INTERACTS", "PERSON"),
        ("PERSON", "OWNS", "HORCRUX"),
        ("ORGANIZATION", "LED_BY", "PERSON"),
    ]

    # Set up the pipeline
    pipe = Pipeline()
    pipe.add_component(PdfLoader(), "pdf_loader")
    pipe.add_component(
        LangChainTextSplitterAdapter(CharacterTextSplitter(separator=". \n")),
        "splitter",
    )
    pipe.add_component(SchemaBuilder(), "schema")
    pipe.add_component(
        LLMEntityRelationExtractor(
            llm=OpenAILLM(
                model_name="gpt-4o",
                model_params={
                    "max_tokens": 2000,
                    "response_format": {"type": "json_object"},
                },
            ),
            on_error=OnError.RAISE,
        ),
        "extractor",
    )
    pipe.add_component(Neo4jWriter(neo4j_driver), "writer")
    pipe.connect("pdf_loader", "splitter", input_config={"text": "pdf_loader.text"})
    pipe.connect("splitter", "extractor", input_config={"chunks": "splitter"})
    pipe.connect(
        "schema",
        "extractor",
        input_config={
            "schema": "schema",
            "document_info": "pdf_loader.document_info",
        },
    )
    pipe.connect(
        "extractor",
        "writer",
        input_config={"graph": "extractor"},
    )

    pipe_inputs = {
        "pdf_loader": {
            "filepath": "examples/pipeline/Harry Potter and the Death Hallows Summary.pdf"
        },
        "schema": {
            "entities": entities,
            "relations": relations,
            "potential_schema": potential_schema,
        },
    }
    return await pipe.run(pipe_inputs)


if __name__ == "__main__":
    with neo4j.GraphDatabase.driver(
        "bolt://localhost:7687", auth=("neo4j", "password")
    ) as driver:
        print(asyncio.run(main(driver)))
