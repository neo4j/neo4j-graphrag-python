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
from neo4j_graphrag.experimental.components.entity_relation_extractor import OnError
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.llm.openai_llm import OpenAILLM

logging.basicConfig(level=logging.INFO)


async def main(neo4j_driver: neo4j.Driver) -> None:
    # Instantiate Entity and Relation objects
    entities = ["PERSON", "ORGANIZATION", "HORCRUX", "LOCATION"]
    relations = ["SITUATED_AT", "INTERACTS", "OWNS", "LED_BY"]
    potential_schema = [
        ("PERSON", "SITUATED_AT", "LOCATION"),
        ("PERSON", "INTERACTS", "PERSON"),
        ("PERSON", "OWNS", "HORCRUX"),
        ("ORGANIZATION", "LED_BY", "PERSON"),
    ]

    # Instantiate the LLM
    llm = OpenAILLM(
        model_name="gpt-4o",
        model_params={
            "max_tokens": 2000,
            "response_format": {"type": "json_object"},
        },
    )

    # Create an instance of the SimpleKGPipeline
    kg_builder_pdf = SimpleKGPipeline(
        llm=llm,
        driver=neo4j_driver,
        entities=entities,
        relations=relations,
        potential_schema=potential_schema,
        from_pdf=True,
        on_error=OnError.RAISE,
    )

    # Run the knowledge graph building process asynchronously
    pdf_file_path = "examples/pipeline/Harry Potter and the Death Hallows Summary.pdf"
    pdf_result = await kg_builder_pdf.run_async(file_path=pdf_file_path)
    print(f"PDF Processing Result: {pdf_result}")

    # Create an instance of the SimpleKGPipeline for text input
    kg_builder_text = SimpleKGPipeline(
        llm=llm,
        driver=neo4j_driver,
        entities=entities,
        relations=relations,
        potential_schema=potential_schema,
        from_pdf=False,
        on_error=OnError.RAISE,
    )

    # Run the knowledge graph building process with text input
    text_input = "John Doe lives in New York City."
    text_result = await kg_builder_text.run_async(text=text_input)
    print(f"Text Processing Result: {text_result}")

    await llm.async_client.close()


if __name__ == "__main__":
    with neo4j.GraphDatabase.driver(
        "bolt://localhost:7687", auth=("neo4j", "password")
    ) as driver:
        asyncio.run(main(driver))
