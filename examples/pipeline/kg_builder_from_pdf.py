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
import asyncio
from typing import Any

import neo4j
from langchain_text_splitters import CharacterTextSplitter
from neo4j_genai.experimental.components.entity_relation_extractor import (
    LLMEntityRelationExtractor,
    OnError,
)
from neo4j_genai.experimental.components.kg_writer import Neo4jWriter
from neo4j_genai.experimental.components.pdf_loader import PdfLoader
from neo4j_genai.experimental.components.schema import (
    SchemaBuilder,
    SchemaEntity,
    SchemaRelation,
)
from neo4j_genai.experimental.components.text_splitters.langchain import (
    LangChainTextSplitterAdapter,
)
from neo4j_genai.experimental.pipeline import Component, DataModel
from neo4j_genai.llm import OpenAILLM


class FuncPipeline:
    def __init__(self):
        self.components = []

    def add(self, component, **kwargs):
        self.components.append((component, kwargs))
        return self

    async def run(self):
        data = None
        for component, kwargs in self.components:
            if isinstance(component, Component):
                data = await component.run(data, **kwargs)
            else:
                raise TypeError("All elements must be instances of Component")
        return data

    def finalise(self):
        print("Pipeline is finalised and ready to run.")

    def __call__(self, component, **kwargs):
        return self.add(component, **kwargs)


class Neo4jGraph(DataModel):
    entities: list[dict[str, Any]]
    relations: list[dict[str, Any]]


async def main(neo4j_driver: neo4j.Driver):
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
    pipeline = FuncPipeline()
    pdf_loader = PdfLoader(
        filepath="examples/pipeline/Harry Potter and the Death Hallows Summary.pdf"
    )
    text_splitter = LangChainTextSplitterAdapter(CharacterTextSplitter())
    schema_builder = SchemaBuilder(
        entities=entities, relations=relations, potential_schema=potential_schema
    )
    er_extractor = LLMEntityRelationExtractor(
        llm=OpenAILLM(
            model_name="gpt-4o",
            model_params={
                "max_tokens": 1000,
                "response_format": {"type": "json_object"},
            },
        ),
        on_error=OnError.RAISE,
    )

    neo4j_writer = Neo4jWriter(neo4j_driver)

    pipeline = pdf_loader(pipeline)
    pipeline = text_splitter(pipeline)
    pipeline = er_extractor(pipeline, schema=schema_builder.schema_config)
    pipeline = neo4j_writer(pipeline)

    pipeline.finalise()
    result = await pipeline.run()
    print(result)


if __name__ == "__main__":
    with neo4j.GraphDatabase.driver(
        "bolt://localhost:7687", auth=("neo4j", "password")
    ) as driver:
        print(asyncio.run(main(driver)))
