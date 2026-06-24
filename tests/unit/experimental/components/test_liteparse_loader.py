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

"""Tests for LiteParseLoader — liteparse is mocked throughout so no install needed."""

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest
from fsspec.implementations.local import LocalFileSystem

from neo4j_graphrag.exceptions import PdfLoaderError
from neo4j_graphrag.experimental.components.liteparse_loader import LiteParseLoader
from neo4j_graphrag.experimental.components.types import DocumentType

SAMPLE_PDF = str(Path(__file__).parent / "sample_data/lorem_ipsum.pdf")
SAMPLE_TEXT = "Lorem ipsum dolor sit amet."


def _make_liteparse_stub(parse_text: str = SAMPLE_TEXT) -> ModuleType:
    result = SimpleNamespace(text=parse_text)
    LiteParse = MagicMock(return_value=MagicMock(parse=MagicMock(return_value=result)))
    stub = ModuleType("liteparse")
    stub.LiteParse = LiteParse  # type: ignore[attr-defined]
    return stub


@pytest.fixture(autouse=True)
def inject_liteparse_stub() -> Generator[ModuleType, None, None]:
    stub = _make_liteparse_stub()
    with patch.dict(sys.modules, {"liteparse": stub}), patch(
        "neo4j_graphrag.experimental.components.liteparse_loader._LiteParse",
        stub.LiteParse,
    ), patch(
        "neo4j_graphrag.experimental.components.liteparse_loader._LITEPARSE_AVAILABLE",
        True,
    ):
        yield stub


def test_load_file_local_fs_returns_text(inject_liteparse_stub: ModuleType) -> None:
    loader = LiteParseLoader()
    text = loader.load_file(SAMPLE_PDF, fs=LocalFileSystem())
    assert text == SAMPLE_TEXT
    inject_liteparse_stub.LiteParse.return_value.parse.assert_called_once_with(
        SAMPLE_PDF
    )


def test_load_file_non_local_fs_uses_bytes(inject_liteparse_stub: ModuleType) -> None:
    fake_fs = MagicMock()
    fake_bytes = b"%PDF fake"
    fake_fs.open.return_value.__enter__ = MagicMock(
        return_value=MagicMock(read=MagicMock(return_value=fake_bytes))
    )
    fake_fs.open.return_value.__exit__ = MagicMock(return_value=False)

    loader = LiteParseLoader()
    with patch(
        "neo4j_graphrag.experimental.components.liteparse_loader.is_default_fs",
        return_value=False,
    ):
        text = loader.load_file(SAMPLE_PDF, fs=fake_fs)

    assert text == SAMPLE_TEXT
    inject_liteparse_stub.LiteParse.return_value.parse.assert_called_once_with(
        fake_bytes
    )


def test_load_file_wraps_parse_error_as_pdf_loader_error(
    inject_liteparse_stub: ModuleType,
) -> None:
    inject_liteparse_stub.LiteParse.return_value.parse.side_effect = RuntimeError(
        "corrupt PDF"
    )
    loader = LiteParseLoader()
    with pytest.raises(PdfLoaderError):
        loader.load_file(SAMPLE_PDF, fs=LocalFileSystem())


def test_missing_liteparse_raises_import_error() -> None:
    with patch(
        "neo4j_graphrag.experimental.components.liteparse_loader._LITEPARSE_AVAILABLE",
        False,
    ):
        loader = LiteParseLoader()
        with pytest.raises(ImportError, match="pip install"):
            loader.load_file(SAMPLE_PDF, fs=LocalFileSystem())


@pytest.mark.asyncio
async def test_run_returns_loaded_document() -> None:
    loader = LiteParseLoader()
    doc = await loader.run(filepath=SAMPLE_PDF)
    assert doc.text == SAMPLE_TEXT
    assert doc.document_info.document_type == DocumentType.PDF
    assert doc.document_info.path == SAMPLE_PDF


@pytest.mark.asyncio
async def test_run_with_path_object() -> None:
    loader = LiteParseLoader()
    doc = await loader.run(filepath=Path(SAMPLE_PDF))
    assert doc.document_info.path == SAMPLE_PDF


@pytest.mark.asyncio
async def test_run_passes_metadata() -> None:
    loader = LiteParseLoader()
    meta = {"source": "test", "lang": "en"}
    doc = await loader.run(filepath=SAMPLE_PDF, metadata=meta)
    assert doc.document_info.metadata == meta


@pytest.mark.asyncio
async def test_run_fs_string_resolves() -> None:
    loader = LiteParseLoader()
    doc = await loader.run(filepath=SAMPLE_PDF, fs="file")
    assert doc.text == SAMPLE_TEXT


def test_constructor_kwargs_forwarded_to_liteparse(
    inject_liteparse_stub: ModuleType,
) -> None:
    loader = LiteParseLoader(
        ocr_enabled=True,
        ocr_language="fra",
        dpi=300,
        target_pages="1-3",
        password="s3cr3t",
    )
    loader.load_file(SAMPLE_PDF, fs=LocalFileSystem())
    inject_liteparse_stub.LiteParse.assert_called_once_with(
        ocr_enabled=True,
        output_format="text",
        ocr_language="fra",
        dpi=300,
        target_pages="1-3",
        password="s3cr3t",
    )


def test_output_format_markdown_forwarded_to_liteparse(
    inject_liteparse_stub: ModuleType,
) -> None:
    loader = LiteParseLoader(output_format="markdown")
    loader.load_file(SAMPLE_PDF, fs=LocalFileSystem())
    inject_liteparse_stub.LiteParse.assert_called_once_with(
        ocr_enabled=False,
        output_format="markdown",
    )
