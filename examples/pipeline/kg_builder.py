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

import neo4j
from langchain_text_splitters import CharacterTextSplitter
from neo4j_genai.components.entity_relation_extractor import LLMEntityRelationExtractor
from neo4j_genai.components.kg_writer import Neo4jWriter
from neo4j_genai.components.text_splitters.langchain import LangChainTextSplitterAdapter
from neo4j_genai.llm import OpenAILLM
from neo4j_genai.pipeline import Component, DataModel, Pipeline

# logging.getLogger(__name__).setLevel(logging.DEBUG)
# logging.basicConfig(level=logging.DEBUG)


class SchemaModel(DataModel):
    data_schema: dict[str, str]


class SchemaBuilder(Component):
    async def run(self, schema: dict[str, str]) -> SchemaModel:
        return SchemaModel(data_schema=schema)


if __name__ == "__main__":
    driver = neo4j.GraphDatabase.driver(
        "bolt://localhost:7687", auth=("neo4j", "password")
    )

    pipe = Pipeline()
    pipe.add_component(
        "splitter", LangChainTextSplitterAdapter(CharacterTextSplitter())
    )
    pipe.add_component("schema", SchemaBuilder())
    pipe.add_component(
        "extractor",
        LLMEntityRelationExtractor(
            llm=OpenAILLM(
                model_name="gpt-4o",
                model_params={
                    "max_tokens": 1000,
                    "response_format": {"type": "json_object"},
                },
            )
        ),
    )
    pipe.add_component("writer", Neo4jWriter(driver))
    pipe.connect("splitter", "extractor", input_config={"chunks": "splitter"})
    pipe.connect("schema", "extractor", input_config={"schema": "schema"})
    pipe.connect(
        "extractor",
        "writer",
        input_config={"graph": "extractor"},
    )

    pipe_inputs = {
        "splitter": {
            "text": """Graphs are everywhere.
            GraphRAG is the future of Artificial Intelligence.
            Robots are already running the world."""
        },
        "schema": {"schema": {}},
    }
    print(asyncio.run(pipe.run(pipe_inputs)))

    driver.close()
