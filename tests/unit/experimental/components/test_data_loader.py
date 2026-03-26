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

from pathlib import Path
from typing import Optional, Union
from unittest.mock import patch

import pytest
from fsspec import AbstractFileSystem
from fsspec.implementations.local import LocalFileSystem
from neo4j_graphrag.exceptions import MarkdownLoadError, PdfLoaderError
from neo4j_graphrag.experimental.components.data_loader import (
    MarkdownLoader,
    PdfLoader,
)
from neo4j_graphrag.experimental.components.types import DocumentType, LoadedDocument

BASE_DIR = Path(__file__).parent


@pytest.fixture
def pdf_loader() -> PdfLoader:
    return PdfLoader()


@pytest.fixture
def dummy_pdf_path() -> str:
    return str(BASE_DIR / "sample_data/lorem_ipsum.pdf")


@pytest.fixture
def dummy_md_path() -> str:
    return str(BASE_DIR / "sample_data/hello.md")


def test_pdf_loading(pdf_loader: PdfLoader, dummy_pdf_path: str) -> None:
    expected_content = "Lorem ipsum dolor sit amet."
    actual_content = pdf_loader.load_file(dummy_pdf_path, fs=LocalFileSystem())
    assert actual_content == expected_content


def test_pdf_processing_error(pdf_loader: PdfLoader, dummy_pdf_path: str) -> None:
    with patch(
        "fsspec.implementations.local.LocalFileSystem.open",
        side_effect=Exception("Failed to open"),
    ):
        with pytest.raises(PdfLoaderError):
            pdf_loader.load_file(dummy_pdf_path, fs=LocalFileSystem())


def test_markdown_processing_error(dummy_md_path: str) -> None:
    with patch(
        "fsspec.implementations.local.LocalFileSystem.open",
        side_effect=Exception("Failed to open"),
    ):
        with pytest.raises(MarkdownLoadError):
            MarkdownLoader.load_file(dummy_md_path, fs=LocalFileSystem())


def test_markdown_loading() -> None:
    md_path = str(BASE_DIR / "sample_data/hello.md")
    text = MarkdownLoader.load_file(md_path, fs=LocalFileSystem())
    assert "# Hello" in text
    assert "Markdown **content**" in text


@pytest.mark.asyncio
async def test_markdown_loader_run() -> None:
    md_path = BASE_DIR / "sample_data/hello.md"
    loader = MarkdownLoader()
    doc = await loader.run(filepath=md_path)
    assert doc.document_info.document_type == DocumentType.MARKDOWN
    assert "# Hello" in doc.text


@pytest.mark.asyncio
async def test_pdf_loader_run() -> None:
    """``PdfLoader.run`` wraps ``load_file`` with :class:`DocumentInfo` (default ``fs``)."""
    pdf_path = BASE_DIR / "sample_data/lorem_ipsum.pdf"
    loader = PdfLoader()
    doc = await loader.run(filepath=pdf_path)
    assert doc.document_info.document_type == DocumentType.PDF
    assert doc.document_info.path == str(pdf_path)
    assert doc.text == "Lorem ipsum dolor sit amet."


@pytest.mark.asyncio
async def test_pdf_loader_run_fs_string_resolves_with_fsspec(
    dummy_pdf_path: str,
) -> None:
    """``fs`` may be a protocol name passed to ``fsspec.filesystem`` (e.g. ``\"file\"``)."""
    loader = PdfLoader()
    doc = await loader.run(filepath=dummy_pdf_path, fs="file")
    assert "Lorem ipsum" in doc.text


@pytest.mark.asyncio
async def test_markdown_loader_run_fs_string() -> None:
    md_path = str(BASE_DIR / "sample_data/hello.md")
    loader = MarkdownLoader()
    doc = await loader.run(filepath=md_path, fs="file")
    assert doc.document_info.document_type == DocumentType.MARKDOWN
    assert "# Hello" in doc.text


@pytest.mark.asyncio
async def test_run_passes_metadata_to_document_info(dummy_pdf_path: str) -> None:
    loader = PdfLoader()
    meta = {"source": "unit-test", "lang": "en"}
    doc = await loader.run(filepath=dummy_pdf_path, metadata=meta)
    assert doc.document_info.metadata == meta


class _PdfLoaderWithDerivedMetadata(PdfLoader):
    """Exercise :meth:`DataLoader.get_document_metadata` override."""

    async def run(
        self,
        filepath: Union[str, Path],
        metadata: Optional[dict[str, str]] = None,
        fs: Optional[Union[AbstractFileSystem, str]] = None,
    ) -> LoadedDocument:
        return await super().run(filepath=filepath, metadata=metadata, fs=fs)

    def get_document_metadata(
        self, text: str, metadata: dict[str, str] | None = None
    ) -> dict[str, str] | None:
        base = dict(metadata or {})
        base["text_length"] = str(len(text))
        return base


@pytest.mark.asyncio
async def test_get_document_metadata_override_merges_into_document_info(
    dummy_pdf_path: str,
) -> None:
    loader = _PdfLoaderWithDerivedMetadata()
    doc = await loader.run(
        filepath=dummy_pdf_path,
        metadata={"source": "derived-test"},
    )
    assert doc.document_info.metadata is not None
    assert doc.document_info.metadata["source"] == "derived-test"
    assert doc.document_info.metadata["text_length"] == str(len(doc.text))


def test_pdf_loader_non_local_filesystem_branch_uses_bytesio(
    dummy_pdf_path: str,
) -> None:
    """Non-\"default\" local FS (``auto_mkdir=True``) reads into BytesIO for pypdf."""
    from neo4j_graphrag.experimental.components.data_loader import is_default_fs

    fs = LocalFileSystem(auto_mkdir=True)
    assert is_default_fs(fs) is False
    text = PdfLoader.load_file(dummy_pdf_path, fs=fs)
    assert text == "Lorem ipsum dolor sit amet."


def test_pdf_loader_backward_compat_reexport_module() -> None:
    """``pdf_loader`` submodule re-exports the same classes as ``data_loader``."""
    from neo4j_graphrag.experimental.components.data_loader import (
        DataLoader as DataLoaderDirect,
        PdfLoader as PdfLoaderDirect,
    )
    from neo4j_graphrag.experimental.components.pdf_loader import (
        DataLoader as DataLoaderReexport,
        PdfLoader as PdfLoaderReexport,
    )

    assert PdfLoaderDirect is PdfLoaderReexport
    assert DataLoaderDirect is DataLoaderReexport
