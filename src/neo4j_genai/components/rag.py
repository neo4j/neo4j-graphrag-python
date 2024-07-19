from __future__ import annotations

import asyncio
from typing import Any

from neo4j_genai.core.pipeline import Component, Pipeline
from neo4j_genai.types import RetrieverResult, RetrieverResultItem


class Retriever(Component):
    def search(self, *args: Any, **kwargs: Any) -> RetrieverResult:
        return RetrieverResult(
            items=[
                RetrieverResultItem(content="my context item 1"),
                RetrieverResultItem(content="my context item 2"),
            ]
        )

    async def process(self, query: str) -> dict[str, Any]:
        res = self.search(query)
        return {"context": "\n".join(c.content for c in res.items)}


class PromptTemplate(Component):
    async def process(self, query: str, context: list[str]) -> dict[str, Any]:
        return {"prompt": f"my prompt using '{context}', query '{query}'"}


class LLM(Component):
    async def process(self, prompt: str) -> dict[str, Any]:
        return {"answer": f"some text based on '{prompt}'"}


if __name__ == "__main__":
    pipe = Pipeline()
    pipe.add_component("retrieve", Retriever())
    pipe.add_component("augment", PromptTemplate())
    pipe.add_component("generate", LLM())
    pipe.connect("retrieve", "augment", {"context": "retrieve.context"})
    pipe.connect("augment", "generate", {"prompt": "augment.prompt"})

    query = "my question"
    print(
        asyncio.run(
            pipe.run({"retrieve": {"query": query}, "augment": {"query": query}})
        )
    )
