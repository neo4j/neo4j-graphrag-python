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
from unittest.mock import MagicMock, Mock, patch

import neo4j
import pytest
from neo4j_graphrag.experimental.components.neo4j_reader import (
    Neo4jChunkReader,
)
from neo4j_graphrag.experimental.components.types import LexicalGraphConfig, TextChunks


@pytest.mark.asyncio
async def test_neo4j_chunk_reader(driver: Mock) -> None:
    driver.execute_query.return_value = (
        [neo4j.Record({"chunk": {"index": 0, "text": "some text", "id": "azerty"}})],
        None,
        None,
    )
    chunk_reader = Neo4jChunkReader(driver)
    res = await chunk_reader.run()

    driver.execute_query.assert_called_once_with(
        "MATCH (c:`Chunk`) RETURN c { .*, id: elementId(c), embedding: null } as chunk ORDER BY c.index"
    )

    assert isinstance(res, TextChunks)
    assert len(res.chunks) == 1
    chunk = res.chunks[0]
    assert chunk.text == "some text"
    assert chunk.index == 0
    assert chunk.metadata == {"id": "azerty"}


@pytest.mark.asyncio
async def test_neo4j_chunk_reader_custom_lg_config(driver: Mock) -> None:
    driver.execute_query.return_value = (
        [
            neo4j.Record(
                {
                    "chunk": {
                        "k": 0,
                        "content": "some text",
                        "id": "azerty",
                        "other": "property",
                    }
                }
            )
        ],
        None,
        None,
    )
    chunk_reader = Neo4jChunkReader(driver)
    res = await chunk_reader.run(
        lexical_graph_config=LexicalGraphConfig(
            chunk_node_label="Page",
            chunk_text_property="content",
            chunk_index_property="k",
        )
    )

    driver.execute_query.assert_called_once_with(
        "MATCH (c:`Page`) RETURN c { .*, id: elementId(c), embedding: null } as chunk ORDER BY c.k"
    )

    assert isinstance(res, TextChunks)
    assert len(res.chunks) == 1
    chunk = res.chunks[0]
    assert chunk.text == "some text"
    assert chunk.index == 0
    assert chunk.metadata == {"id": "azerty", "other": "property"}


@pytest.mark.asyncio
async def test_neo4j_chunk_reader_do_not_fetch_embedding(driver: Mock) -> None:
    driver.execute_query.return_value = (
        [
            neo4j.Record(
                {
                    "chunk": {
                        "index": 0,
                        "text": "some text",
                        "other": "property",
                        "embedding": [1.0, 2.0, 3.0],
                        "id": "azerty",
                    }
                }
            )
        ],
        None,
        None,
    )
    chunk_reader = Neo4jChunkReader(driver, fetch_embeddings=True)
    res = await chunk_reader.run()

    driver.execute_query.assert_called_once_with(
        "MATCH (c:`Chunk`) RETURN c { .*, id: elementId(c) } as chunk ORDER BY c.index"
    )

    assert isinstance(res, TextChunks)
    assert len(res.chunks) == 1
    chunk = res.chunks[0]
    assert chunk.text == "some text"
    assert chunk.index == 0
    assert chunk.metadata == {
        "other": "property",
        "id": "azerty",
        "embedding": [1.0, 2.0, 3.0],
    }
