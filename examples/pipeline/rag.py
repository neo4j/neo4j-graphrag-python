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
"""This example illustrates how to use a Pipeline with
the existing Retriever and LLM interfaces. It consists
in creating a Component wrapper around the required
objects.
"""

from __future__ import annotations

import asyncio
from typing import List

import neo4j
from neo4j_genai.embeddings.openai import OpenAIEmbeddings
from neo4j_genai.experimental.pipeline import Component, Pipeline
from neo4j_genai.experimental.pipeline.component import DataModel
from neo4j_genai.experimental.pipeline.pipeline import PipelineResult
from neo4j_genai.experimental.pipeline.types import (
    ComponentConfig,
    ConnectionConfig,
    PipelineConfig,
)
from neo4j_genai.generation import PromptTemplate, RagTemplate
from neo4j_genai.llm import LLMInterface, OpenAILLM
from neo4j_genai.retrievers import VectorRetriever
from neo4j_genai.retrievers.base import Retriever


class ComponentResultDataModel(DataModel):
    """A simple DataModel with a single text field"""

    text: str


class RetrieverComponent(Component):
    def __init__(self, retriever: Retriever) -> None:
        self.retriever = retriever

    async def run(self, query: str) -> ComponentResultDataModel:
        res = self.retriever.search(query_text=query)
        return ComponentResultDataModel(text="\n".join(c.content for c in res.items))


class PromptTemplateComponent(Component):
    def __init__(self, prompt: PromptTemplate) -> None:
        self.prompt = prompt

    async def run(self, query: str, context: List[str]) -> ComponentResultDataModel:
        prompt = self.prompt.format(query, context, examples="")
        return ComponentResultDataModel(text=prompt)


class LLMComponent(Component):
    def __init__(self, llm: LLMInterface) -> None:
        self.llm = llm

    async def run(self, prompt: str) -> ComponentResultDataModel:
        llm_response = self.llm.invoke(prompt)
        return ComponentResultDataModel(text=llm_response.content)


if __name__ == "__main__":
    driver = neo4j.GraphDatabase.driver(
        "bolt://localhost:7687",
        auth=("neo4j", "password"),
        database="neo4j",
    )
    embedder = OpenAIEmbeddings()
    retriever = VectorRetriever(
        driver, index_name="moviePlotsEmbedding", embedder=embedder
    )
    prompt_template = RagTemplate()
    llm = OpenAILLM(model_name="gpt-4o")

    pipe = Pipeline.from_template(
        PipelineConfig(
            components=[
                ComponentConfig(
                    name="retrieve", component=RetrieverComponent(retriever)
                ),
                ComponentConfig(
                    name="augment", component=PromptTemplateComponent(prompt_template)
                ),
                ComponentConfig(name="generate", component=LLMComponent(llm)),
            ],
            connections=[
                ConnectionConfig(
                    start="retrieve",
                    end="augment",
                    input_config={"context": "retrieve.text"},
                ),
                ConnectionConfig(
                    start="augment",
                    end="generate",
                    input_config={"prompt": "augment.text"},
                ),
            ],
        )
    )

    query = "A movie about the US presidency"
    pipe_output: PipelineResult = asyncio.run(
        pipe.run({"retrieve": {"query": query}, "augment": {"query": query}})
    )
    print(pipe_output.result["generate"]["text"])

    driver.close()
