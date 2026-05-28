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
"""Document loaders: base class, PDF, Markdown, LiteParse, and extension-based dispatch."""

from __future__ import annotations

import io
from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Union, cast

import fsspec
import pypdf
from fsspec import AbstractFileSystem
from fsspec.implementations.local import LocalFileSystem

from neo4j_graphrag.exceptions import MarkdownLoadError, PdfLoaderError

try:
    from liteparse import LiteParse as _LiteParse  # type: ignore[import]

    _LITEPARSE_AVAILABLE = True
except ImportError:
    _LiteParse = None  # type: ignore[assignment,misc]
    _LITEPARSE_AVAILABLE = False
from neo4j_graphrag.experimental.components.types import (
    DocumentInfo,
    DocumentType,
    LoadedDocument,
)
from neo4j_graphrag.experimental.pipeline.component import Component


def is_default_fs(fs: fsspec.AbstractFileSystem) -> bool:
    return isinstance(fs, LocalFileSystem) and not fs.auto_mkdir


class DataLoader(Component):
    """
    Interface for loading data of various input types.
    """

    def get_document_metadata(
        self, text: str, metadata: Optional[Dict[str, str]] = None
    ) -> Dict[str, str] | None:
        return metadata

    @abstractmethod
    async def run(
        self,
        filepath: Union[str, Path],
        metadata: Optional[Dict[str, str]] = None,
    ) -> LoadedDocument: ...


class PdfLoader(DataLoader):
    """Loads text from PDF files using pypdf."""

    @staticmethod
    def load_file(
        file: str,
        fs: AbstractFileSystem,
    ) -> str:
        """Parse a PDF file and return extracted text."""
        try:
            with fs.open(file, "rb") as fp:
                stream = fp if is_default_fs(fs) else io.BytesIO(fp.read())
                pdf = pypdf.PdfReader(stream)
                num_pages = len(pdf.pages)
                text_parts = (
                    pdf.pages[page].extract_text() for page in range(num_pages)
                )
                return "\n".join(text_parts)
        except Exception as e:
            raise PdfLoaderError(e)

    async def run(
        self,
        filepath: Union[str, Path],
        metadata: Optional[Dict[str, str]] = None,
        fs: Optional[Union[AbstractFileSystem, str]] = None,
    ) -> LoadedDocument:
        if not isinstance(filepath, str):
            filepath = str(filepath)
        if isinstance(fs, str):
            fs = fsspec.filesystem(fs)
        elif fs is None:
            fs = LocalFileSystem()
        text = self.load_file(filepath, fs)
        return LoadedDocument(
            text=text,
            document_info=DocumentInfo(
                path=filepath,
                metadata=self.get_document_metadata(text, metadata),
                document_type=DocumentType.PDF,
            ),
        )


class MarkdownLoader(DataLoader):
    """Loads UTF-8 Markdown (``.md`` / ``.markdown``) files as plain text."""

    @staticmethod
    def load_file(
        file: str,
        fs: AbstractFileSystem,
    ) -> str:
        try:
            with fs.open(file, "rb") as fp:
                raw = fp.read()
            return cast(str, raw.decode("utf-8"))
        except Exception as e:
            raise MarkdownLoadError(e)

    async def run(
        self,
        filepath: Union[str, Path],
        metadata: Optional[Dict[str, str]] = None,
        fs: Optional[Union[AbstractFileSystem, str]] = None,
    ) -> LoadedDocument:
        if not isinstance(filepath, str):
            filepath = str(filepath)
        if isinstance(fs, str):
            fs = fsspec.filesystem(fs)
        elif fs is None:
            fs = LocalFileSystem()
        text = MarkdownLoader.load_file(filepath, fs)
        return LoadedDocument(
            text=text,
            document_info=DocumentInfo(
                path=filepath,
                metadata=self.get_document_metadata(text, metadata),
                document_type=DocumentType.MARKDOWN,
            ),
        )


class LiteParseLoader(DataLoader):
    """Loads and parses PDF files using LiteParse (local, no cloud dependency).

    LiteParse uses a compiled Rust core and optional Tesseract OCR to extract text
    from PDFs, DOCX, XLSX, PPTX, and images.  It runs fully offline — no API key
    or network access required.

    Requires the ``liteparse`` optional extra::

        pip install "neo4j-graphrag[liteparse]"

    Args:
        ocr_enabled: Enable Tesseract OCR for scanned pages.
        ocr_server_url: URL of a remote OCR server (optional).
        ocr_language: Tesseract language code, e.g. ``"eng"``, ``"fra"``.
        dpi: Rendering resolution for OCR (default 300).
        target_pages: Page range string, e.g. ``"1-5,10,15-20"``.
        password: Password for encrypted PDFs.
    """

    def __init__(
        self,
        ocr_enabled: bool = False,
        ocr_server_url: Optional[str] = None,
        ocr_language: Optional[str] = None,
        dpi: Optional[int] = None,
        target_pages: Optional[str] = None,
        password: Optional[str] = None,
    ) -> None:
        optional = {
            "ocr_server_url": ocr_server_url,
            "ocr_language": ocr_language,
            "dpi": dpi,
            "target_pages": target_pages,
            "password": password,
        }
        self._kwargs: Dict[str, Any] = {
            "ocr_enabled": ocr_enabled,
            **{k: v for k, v in optional.items() if v is not None},
        }
        self._parser: Any = None  # lazily initialised

    def _get_parser(self) -> Any:
        if not _LITEPARSE_AVAILABLE:
            raise ImportError(
                "liteparse is required for LiteParseLoader. "
                'Install it with: pip install "neo4j-graphrag[liteparse]"'
            )
        if self._parser is None:
            self._parser = _LiteParse(**self._kwargs)
        return self._parser

    def load_file(self, filepath: str, fs: AbstractFileSystem) -> str:
        """Parse a document and return the full extracted text."""
        parser = self._get_parser()  # ImportError propagates; not caught below
        try:
            if is_default_fs(fs):
                result = parser.parse(filepath)
            else:
                with fs.open(filepath, "rb") as fp:
                    result = parser.parse(fp.read())
            return str(result.text)
        except Exception as e:
            raise PdfLoaderError(e) from e

    async def run(
        self,
        filepath: Union[str, Path],
        metadata: Optional[Dict[str, str]] = None,
        fs: Optional[Union[AbstractFileSystem, str]] = None,
    ) -> LoadedDocument:
        if not isinstance(filepath, str):
            filepath = str(filepath)
        if isinstance(fs, str):
            fs = fsspec.filesystem(fs)
        elif fs is None:
            fs = LocalFileSystem()
        text = self.load_file(filepath, fs)
        return LoadedDocument(
            text=text,
            document_info=DocumentInfo(
                path=filepath,
                metadata=self.get_document_metadata(text, metadata),
                document_type=DocumentType.PDF,
            ),
        )
