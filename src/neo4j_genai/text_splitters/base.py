from __future__ import annotations

from abc import abstractmethod
from typing import Any, Optional

from pydantic import BaseModel

from neo4j_genai.core.component import Component, DataModel


class TextChunk(BaseModel):
    """A chunk of text split from a document by a text splitter.

    Attributes:
        text (str): The raw chunk text.
        metadata (Optional[dict[str, Any]]): Metadata associated with this chunk such as the id of the next chunk in the original document.
    """

    text: str
    metadata: Optional[dict[str, Any]] = None


class TextChunks(DataModel):
    """A collection of text chunks returned from a text splitter.

    Attributes:
        chunks (list[TextChunk]): A list of text chunks.
    """

    chunks: list[TextChunk]


class TextSplitter(Component):
    """Interface for a text splitter."""

    @abstractmethod
    async def run(self, text: str) -> TextChunks:
        pass
