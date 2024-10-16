from typing import Any, Optional

from neo4j_graphrag.experimental.components.entity_relation_extractor import (
    EntityRelationExtractor,
    OnError,
)
from neo4j_graphrag.experimental.components.pdf_loader import DocumentInfo
from neo4j_graphrag.experimental.components.types import Neo4jGraph, TextChunks


class MyExtractor(EntityRelationExtractor):
    def __init__(
        self,
        *args: Any,
        on_error: OnError = OnError.IGNORE,
        create_lexical_graph: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            *args,
            on_error=on_error,
            create_lexical_graph=create_lexical_graph,
            **kwargs,
        )

    async def run(
        self,
        chunks: TextChunks,
        document_info: Optional[DocumentInfo] = None,
        **kwargs: Any,
    ) -> Neo4jGraph:
        # Implement your logic here
        # you can loop over all text chunks with:
        for chunk in chunks.chunks:
            pass
        return Neo4jGraph(nodes=[], relationships=[])
