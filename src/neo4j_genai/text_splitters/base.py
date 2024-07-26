from abc import abstractmethod

from neo4j_genai.core.component import Component, DataModel


class TextChunks(DataModel):
    chunks: list[str]


class TextSplitter(Component):
    @abstractmethod
    async def run(self, text: str) -> TextChunks:
        pass
