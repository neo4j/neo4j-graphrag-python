from typing import Any

from neo4j_genai.types import RetrieverResult, RetrieverResultItem
from neo4j_genai.core.pipeline import Component, Pipeline


class Retriever(Component):
    def search(self, *args: Any, **kwargs: Any) -> RetrieverResult:
        return RetrieverResult(
            items=[
                RetrieverResultItem(content="my context item 1"),
                RetrieverResultItem(content="my context item 2"),
            ]
        )

    def run(self, query: str):
        res = self.search(query)
        return {"context": "\n".join(c.content for c in res.items)}


class PromptTemplate(Component):
    def run(self, query: str, context: list):
        return {"prompt": f"my prompt using '{context}', query '{query}'"}


class LLM(Component):
    def run(self, prompt):
        return {"answer": f"some text based on '{prompt}'"}


if __name__ == "__main__":
    pipe = Pipeline()
    pipe.add_component(Retriever(), "retrieve")
    pipe.add_component(PromptTemplate(), "augment", {"context": "retrieve.context"})
    pipe.add_component(LLM(), "generate", {"prompt": "augment.prompt"})

    query = "my qestion"
    print(pipe.run({"retrieve": {"query": query}, "augment": {"query": query}}))
