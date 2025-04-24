from typing import Any

# from neo4j_graphrag.retrievers.base import Retriever
from neo4j_graphrag.types import RawSearchResult, RetrieverResult, RetrieverResultItem


class MCPServerInterface:
    def __init__(self, *args, **kwargs):
        pass

    def get_tools(self):
        return []

    def execute_tool(self, tool) -> Any:
        return ""


class MCPRetriever:
    def __init__(self, server: MCPServerInterface) -> None:
        super().__init__()
        self.server = server
        self.tools = server.get_tools()

    def search(self, query_text: str) -> RetrieverResult:
        """Reimplement the search method because we can't inherit from
        the Retriever interface (no need for neo4j.driver here).

        1. Call llm with a list of tools
        2. Call MCP server for specific tool and LLM-generated arguments
        3. Return all results as context in RetrieverResult
        """
        raw_result = RawSearchResult(records=[])
        search_items = [RetrieverResultItem(content=str(record)) for record in raw_result.records]
        metadata = raw_result.metadata or {}
        metadata["__retriever"] = self.__class__.__name__
        metadata["__tool_results"] = {}
        return RetrieverResult(
            items=search_items,
            metadata=metadata,
        )
