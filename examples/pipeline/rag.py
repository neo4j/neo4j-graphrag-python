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
from typing import Any

from neo4j_genai.pipeline import Component, Pipeline
from neo4j_genai.pipeline.component import DataModel
from neo4j_genai.pipeline.types import ComponentConfig, ConnectionConfig, PipelineConfig
from neo4j_genai.types import RetrieverResult, RetrieverResultItem


class RetrieverOutputMessage(DataModel):
    context: str


class Retriever(Component):
    output_data_model = RetrieverOutputMessage

    def search(self, *args: Any, **kwargs: Any) -> RetrieverResult:
        return RetrieverResult(
            items=[
                RetrieverResultItem(content="my context item 1"),
                RetrieverResultItem(content="my context item 2"),
            ]
        )

    async def run(self, query: str) -> RetrieverOutputMessage:
        res = self.search(query)
        return RetrieverOutputMessage(
            **{"context": "\n".join(c.content for c in res.items)}
        )


class PromptTemplateOutput(DataModel):
    prompt: str


class PromptTemplate(Component):
    def __init__(self, param: str = "default") -> None:
        self.param = param

    async def run(self, query: str, context: list[str]) -> PromptTemplateOutput:
        return PromptTemplateOutput(
            **{
                "prompt": f"my prompt using '{context}' and '{self.param}', query '{query}'"
            }
        )


class LLMOutputMessage(DataModel):
    answer: str


class LLM(Component):
    async def run(self, prompt: str) -> LLMOutputMessage:
        return LLMOutputMessage.model_validate(
            {"answer": f"some text based on '{prompt}'"}
        )


if __name__ == "__main__":
    # retriever = Retriever()
    # print(asyncio.run(retriever.run("my context item 1")))

    pipe = Pipeline.from_template(
        PipelineConfig(
            components=[
                ComponentConfig(name="retrieve", component=Retriever()),
                ComponentConfig(name="augment", component=PromptTemplate()),
                ComponentConfig(name="generate", component=LLM()),
            ],
            connections=[
                ConnectionConfig(
                    start="retrieve",
                    end="augment",
                    input_config={"context": "retrieve.context"},
                ),
                ConnectionConfig(
                    start="augment",
                    end="generate",
                    # input_config={},
                    input_config={"prompt": "augment.prompt"},
                ),
            ],
        )
    )

    query = "my question"
    print(
        asyncio.run(
            pipe.run({"retrieve": {"query": query}, "augment": {"query": query}})
        )
    )

    pipe.set_component("augment", PromptTemplate(param="my param"))
    pipe.reinitialize()
    # print(pipe.show_as_dict())
    print(
        asyncio.run(
            pipe.run({"retrieve": {"query": query}, "augment": {"query": query}})
        )
    )
