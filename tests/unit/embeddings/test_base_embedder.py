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
from unittest.mock import AsyncMock, MagicMock

import pytest
from neo4j_graphrag.embeddings.base import Embedder, _split_into_groups


class _LengthEmbedder(Embedder):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[str] = []

    def embed_query(self, text: str) -> list[float]:
        self.calls.append(text)
        return [float(len(text))]


def test_default_embed_documents_calls_embed_query_per_text() -> None:
    embedder = _LengthEmbedder()
    assert embedder.embed_documents(["a", "bbb", "cc"]) == [[1.0], [3.0], [2.0]]
    assert embedder.calls == ["a", "bbb", "cc"]


def test_default_embed_documents_empty() -> None:
    embedder = _LengthEmbedder()
    assert embedder.embed_documents([]) == []
    assert embedder.calls == []


@pytest.mark.asyncio
async def test_default_async_embed_documents_uses_async_embed_query() -> None:
    embedder = _LengthEmbedder()
    embedder.async_embed_query = AsyncMock(side_effect=[[1.0], [2.0]])  # type: ignore[method-assign]
    embedder.embed_query = MagicMock()  # type: ignore[method-assign]
    assert await embedder.async_embed_documents(["x", "yy"]) == [[1.0], [2.0]]
    assert embedder.async_embed_query.await_count == 2
    embedder.embed_query.assert_not_called()


@pytest.mark.parametrize(
    "texts, size, expected",
    [
        ([], 2, []),
        ([], None, []),
        (["a", "b", "c"], None, [["a", "b", "c"]]),
        (["a", "b", "c"], 3, [["a", "b", "c"]]),
        (["a", "b", "c"], 2, [["a", "b"], ["c"]]),
        (["a", "b", "c"], 1, [["a"], ["b"], ["c"]]),
    ],
)
def test_split_into_groups(
    texts: list[str], size: int | None, expected: list[list[str]]
) -> None:
    assert _split_into_groups(texts, size) == expected
