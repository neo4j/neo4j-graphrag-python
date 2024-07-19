from langchain_text_splitters import TextSplitter as LangChainTextSplitter
from llama_index.core.node_parser import TextSplitter as LlamaIndexTextSplitter

from neo4j_genai.core.pipeline import Component


class TextSplitterAdapter(Component):
    """Adapter for LangChain and LlamaIndex TextSplitters.
    Allows instances of these classes to be used in the knowledge graph builder pipeline.
    """

    def __init__(
        self, text_splitter: LangChainTextSplitter | LlamaIndexTextSplitter
    ) -> None:
        self.text_splitter = text_splitter

    async def run(self, text: str) -> dict[str, list[str]]:
        """
        Splits text into chunks.

        Args:
            text (str): The text to split.

        Returns:
            dict[str, list[str]]: A dictionary with a single key "text_chunks" containing the split text chunks.
        """
        return {"text_chunks": self.text_splitter.split_text(text)}
