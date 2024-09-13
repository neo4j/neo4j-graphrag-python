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
import pytest
from llama_index.core.node_parser.text.sentence import SentenceSplitter
from neo4j_graphrag.experimental.components.text_splitters.llamaindex import (
    LlamaIndexTextSplitterAdapter,
)
from neo4j_graphrag.experimental.components.types import TextChunk, TextChunks

text = """
Lorem ipsum dolor sit amet, consectetur adipiscing elit.
In cursus erat quis ornare condimentum. Ut sollicitudin libero nec quam vestibulum, non tristique augue tempor.
Nulla fringilla, augue ac fermentum ultricies, mauris tellus tempor orci, at tincidunt purus arcu vitae nisl.
Nunc suscipit neque vitae ipsum viverra, eu interdum tortor iaculis.
Suspendisse sit amet quam non ipsum molestie euismod finibus eu nisi. Quisque sit amet aliquet leo, vel auctor dolor.
Sed auctor enim at tempus eleifend. Suspendisse potenti. Suspendisse congue tellus id justo bibendum, at commodo sapien porta.
Nam sagittis nisl vitae nibh pellentesque, et convallis turpis ultrices.
"""


@pytest.mark.asyncio
async def test_llamaindex_adapter() -> None:
    text_splitter = LlamaIndexTextSplitterAdapter(SentenceSplitter())
    text_chunks = await text_splitter.run(text)
    assert isinstance(text_chunks, TextChunks)
    for text_chunk in text_chunks.chunks:
        assert isinstance(text_chunk, TextChunk)
        assert text_chunk.text in text
