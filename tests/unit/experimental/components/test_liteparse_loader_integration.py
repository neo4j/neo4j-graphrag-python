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

"""Integration tests for LiteParseLoader — require ``pip install neo4j-graphrag[liteparse]``.

These tests call the real liteparse library against actual PDF files.
They are skipped automatically when liteparse is not installed.
"""

from pathlib import Path

import pytest
from fsspec.implementations.local import LocalFileSystem
from fsspec.implementations.memory import MemoryFileSystem

from neo4j_graphrag.experimental.components.types import DocumentType

pytest.importorskip(
    "liteparse", reason="liteparse not installed — skipping integration tests"
)

from neo4j_graphrag.experimental.components.data_loader import LiteParseLoader  # noqa: E402

SAMPLE_PDF = Path(__file__).parent / "sample_data/lorem_ipsum.pdf"


def test_real_parse_local_file() -> None:
    """LiteParseLoader extracts text from the sample PDF via the local FS path."""
    loader = LiteParseLoader()
    text = loader.load_file(str(SAMPLE_PDF), fs=LocalFileSystem())
    assert len(text) > 0
    assert "Lorem" in text


def test_real_parse_from_bytes() -> None:
    """Loader works when bytes are read through a non-default FS (bytes branch)."""
    pdf_bytes = SAMPLE_PDF.read_bytes()
    mfs = MemoryFileSystem()
    with mfs.open("/lorem.pdf", "wb") as fh:
        fh.write(pdf_bytes)

    loader = LiteParseLoader()
    text = loader.load_file("/lorem.pdf", fs=mfs)
    assert "Lorem" in text


@pytest.mark.asyncio
async def test_real_run_returns_loaded_document() -> None:
    loader = LiteParseLoader()
    doc = await loader.run(filepath=SAMPLE_PDF)
    assert doc.document_info.document_type == DocumentType.PDF
    assert doc.document_info.path == str(SAMPLE_PDF)
    assert "Lorem" in doc.text


@pytest.mark.asyncio
async def test_real_run_with_metadata() -> None:
    loader = LiteParseLoader()
    meta = {"source": "integration-test"}
    doc = await loader.run(filepath=SAMPLE_PDF, metadata=meta)
    assert doc.document_info.metadata == meta


@pytest.mark.asyncio
async def test_real_run_text_is_nonempty() -> None:
    loader = LiteParseLoader()
    doc = await loader.run(filepath=SAMPLE_PDF)
    assert doc.text.strip() != ""
