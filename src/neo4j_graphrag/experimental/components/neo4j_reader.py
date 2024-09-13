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

from typing import Any, Dict, List, Optional, Union

import neo4j
from pydantic import ConfigDict

from neo4j_graphrag.experimental.components.types import TextChunk, TextChunks
from neo4j_graphrag.experimental.pipeline import Component, DataModel
from tests.unit.experimental.components.lexical_graph import LexicalGraphConfig


class Neo4jRecords(DataModel):
    records: List[neo4j.Record]

    model_config = ConfigDict(arbitrary_types_allowed=True)


class Neo4jReader(Component):
    def __init__(self, driver: neo4j.driver):
        self.driver = driver

    async def run(
        self, query: str, parameters: Optional[Dict[str, Any]] = None
    ) -> Neo4jRecords:
        if isinstance(self.driver, neo4j.AsyncDriver):
            result = await self.driver.execute_query(query, parameters_=parameters)
        else:
            result = self.driver.execute_query(query, parameters_=parameters)
        return Neo4jRecords(records=result.records)


class Neo4jChunkReader(Component):
    def __init__(
        self,
        driver: Union[neo4j.AsyncDriver, neo4j.Driver],
        fetch_embeddings: bool = False,
    ):
        self.reader = Neo4jReader(driver)
        self.fetch_embeddings = fetch_embeddings

    def _get_query(
        self, chunk_label: str, embedding_property: str, index_property: str
    ) -> str:
        if self.fetch_embeddings:
            return (
                f"MATCH (c:`{chunk_label}`) "
                "RETURN c {.*} as chunk "
                f"ORDER BY c.{index_property}"
            )
        return (
            f"MATCH (c:`{chunk_label}`) "
            f"RETURN c {{ .*, {embedding_property}: null }} as chunk "
            f"ORDER BY c.{index_property}"
        )

    async def run(
        self,
        lexical_graph_config: LexicalGraphConfig = LexicalGraphConfig(),
    ) -> TextChunks:
        query = self._get_query(
            lexical_graph_config.chunk_node_label,
            lexical_graph_config.chunk_embedding_property,
            lexical_graph_config.chunk_index_property,
        )
        result = await self.reader.run(query)
        chunks = []
        for record in result.records:
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
