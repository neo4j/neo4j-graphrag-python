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
from unittest.mock import MagicMock, patch

import neo4j
import pytest
from neo4j_graphrag.experimental.components.neo4j_reader import (
    Neo4jChunkReader,
    Neo4jRecords,
)
from neo4j_graphrag.experimental.components.types import TextChunks

from tests.unit.experimental.components.lexical_graph import LexicalGraphConfig


@patch("neo4j_graphrag.experimental.components.neo4j_reader.Neo4jReader.run")
@pytest.mark.asyncio
async def test_neo4j_chunk_reader(mock_reader: MagicMock, driver: neo4j.Driver) -> None:
    mock_reader.return_value = Neo4jRecords(
        records=[
            neo4j.Record(
                {"chunk": {"index": 0, "text": "some text", "other": "property"}}
            )
        ]
    )
    chunk_reader = Neo4jChunkReader(driver)
    res = await chunk_reader.run()

    mock_reader.assert_awaited_once_with(
        "MATCH (c:`Chunk`) RETURN c { .*, embedding: null } as chunk ORDER BY c.index"
    )

    assert isinstance(res, TextChunks)
    assert len(res.chunks) == 1
    chunk = res.chunks[0]
    assert chunk.text == "some text"
    assert chunk.index == 0
    assert chunk.metadata == {"other": "property"}


@patch("neo4j_graphrag.experimental.components.neo4j_reader.Neo4jReader.run")
@pytest.mark.asyncio
async def test_neo4j_chunk_reader_custom_lg_config(
    mock_reader: MagicMock, driver: neo4j.Driver
) -> None:
    mock_reader.return_value = Neo4jRecords(
        records=[
            neo4j.Record(
                {"chunk": {"k": 0, "content": "some text", "other": "property"}}
            )
        ]
    )
    chunk_reader = Neo4jChunkReader(driver)
    res = await chunk_reader.run(
        lexical_graph_config=LexicalGraphConfig(
            chunk_node_label="Page",
            chunk_text_property="content",
            chunk_index_property="k",
        )
    )

    mock_reader.assert_awaited_once_with(
        "MATCH (c:`Page`) RETURN c { .*, embedding: null } as chunk ORDER BY c.index"
    )

    assert isinstance(res, TextChunks)
    assert len(res.chunks) == 1
    chunk = res.chunks[0]
    assert chunk.text == "some text"
    assert chunk.index == 0
    assert chunk.metadata == {"other": "property"}


@patch("neo4j_graphrag.experimental.components.neo4j_reader.Neo4jReader.run")
@pytest.mark.asyncio
async def test_neo4j_chunk_reader_do_not_fetch_embedding(
    mock_reader: MagicMock, driver: neo4j.Driver
) -> None:
    mock_reader.return_value = Neo4jRecords(
        records=[
            neo4j.Record(
                {
                    "chunk": {
                        "index": 0,
                        "text": "some text",
                        "other": "property",
                        "embedding": [1.0, 2.0, 3.0],
                    }
                }
            )
        ]
    )
    chunk_reader = Neo4jChunkReader(driver, fetch_embeddings=True)
    res = await chunk_reader.run()

    mock_reader.assert_awaited_once_with(
        "MATCH (c:`Chunk`) RETURN c {.*} as chunk ORDER BY c.index"
    )

    assert isinstance(res, TextChunks)
    assert len(res.chunks) == 1
    chunk = res.chunks[0]
    assert chunk.text == "some text"
    assert chunk.index == 0
    assert chunk.metadata == {"other": "property", "embedding": [1.0, 2.0, 3.0]}
