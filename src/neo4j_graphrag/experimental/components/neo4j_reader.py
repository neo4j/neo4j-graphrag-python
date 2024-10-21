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

from typing import Union

import neo4j
from pydantic import validate_call

from neo4j_graphrag.experimental.components.types import (
    LexicalGraphConfig,
    TextChunk,
    TextChunks,
)
from neo4j_graphrag.experimental.pipeline import Component
from neo4j_graphrag.utils import execute_query


class Neo4jChunkReader(Component):
    def __init__(
        self,
        driver: Union[neo4j.AsyncDriver, neo4j.Driver],
        fetch_embeddings: bool = False,
    ):
        self.driver = driver
        self.fetch_embeddings = fetch_embeddings

    def _get_query(
        self, chunk_label: str, embedding_property: str, index_property: str
    ) -> str:
        return_properties = [".*", "id: elementId(c)"]
        if not self.fetch_embeddings:
            return_properties.append(f"{embedding_property}: null")
        return (
            f"MATCH (c:`{chunk_label}`) "
            f"RETURN c {{ { ', '.join(return_properties) } }} as chunk "
            f"ORDER BY c.{index_property}"
        )

    @validate_call
    async def run(
        self,
        lexical_graph_config: LexicalGraphConfig = LexicalGraphConfig(),
    ) -> TextChunks:
        query = self._get_query(
            lexical_graph_config.chunk_node_label,
            lexical_graph_config.chunk_embedding_property,
            lexical_graph_config.chunk_index_property,
        )
        result = await execute_query(self.driver, query)
        chunks = []
        for record in result:
            chunk = record.get("chunk")
            text = chunk.pop(lexical_graph_config.chunk_text_property, "")
            index = chunk.pop(lexical_graph_config.chunk_index_property, None)
            chunks.append(
                TextChunk(
                    text=text,
                    index=index,
                    metadata=chunk,
                )
            )
        return TextChunks(chunks=chunks)
