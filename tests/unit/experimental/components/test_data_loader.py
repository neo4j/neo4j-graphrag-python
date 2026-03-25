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
from unittest.mock import patch

import pytest
from fsspec.implementations.local import LocalFileSystem
from neo4j_graphrag.exceptions import PdfLoaderError, UnsupportedDocumentFormatError
from neo4j_graphrag.experimental.components.data_loader import (
    FileLoader,
    MarkdownLoader,
    PdfLoader,
)

BASE_DIR = Path(__file__).parent


@pytest.fixture
def pdf_loader() -> PdfLoader:
    return PdfLoader()


@pytest.fixture
def dummy_pdf_path() -> str:
    return str(BASE_DIR / "sample_data/lorem_ipsum.pdf")


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
    assert doc.document_info.document_type == "markdown"
    assert "# Hello" in doc.text


def test_file_loader_dispatch_pdf() -> None:
    pdf_path = str(BASE_DIR / "sample_data/lorem_ipsum.pdf")
    text = FileLoader.load_file(pdf_path, fs=LocalFileSystem())
    assert text == "Lorem ipsum dolor sit amet."


def test_file_loader_dispatch_markdown() -> None:
    md_path = str(BASE_DIR / "sample_data/hello.md")
    text = FileLoader.load_file(md_path, fs=LocalFileSystem())
    assert "# Hello" in text


def test_file_loader_unsupported_extension() -> None:
    with pytest.raises(UnsupportedDocumentFormatError):
        FileLoader.load_file("/tmp/foo.txt", fs=LocalFileSystem())


@pytest.mark.asyncio
async def test_file_loader_run_sets_document_type() -> None:
    loader = FileLoader()
    md_path = BASE_DIR / "sample_data/hello.md"
    doc = await loader.run(filepath=md_path)
    assert doc.document_info.document_type == "markdown"

    pdf_path = BASE_DIR / "sample_data/lorem_ipsum.pdf"
    doc_pdf = await loader.run(filepath=pdf_path)
    assert doc_pdf.document_info.document_type == "pdf"
