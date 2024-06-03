from typing import Optional
from ..retrievers.base import Retriever
from .llm import LLMInterface
from .prompts import RagTemplate


class RAG:
    def __init__(
        self,
        retriever: Retriever,
        llm: LLMInterface,
        prompt_template: RagTemplate = RagTemplate(),
    ):
        self.retriever = retriever
        self.llm = llm
        self.prompt_template = prompt_template

    def search(self, query: str, retriever_config: Optional[dict] = None) -> str:
        """
        Args:
            query (str): The user question
        """
        retriever_config = retriever_config or {}
        retriever_result = self.retriever.search(query_text=query, **retriever_config)
        context = "\n".join(item.content for item in retriever_result.items)
        prompt = self.prompt_template.format(query=query, context=context)
        return self.llm.invoke(prompt)
