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
"""LiteParse-backed document loader (optional extra: ``neo4j-graphrag[liteparse]``)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import fsspec
from fsspec import AbstractFileSystem
from fsspec.implementations.local import LocalFileSystem

from neo4j_graphrag.exceptions import PdfLoaderError
from neo4j_graphrag.experimental.components.data_loader import DataLoader, is_default_fs
from neo4j_graphrag.experimental.components.types import (
    DocumentInfo,
    DocumentType,
    LoadedDocument,
)

try:
    from liteparse import LiteParse as _LiteParse

    _LITEPARSE_AVAILABLE = True
except ImportError:
    _LiteParse = None  # type: ignore[assignment,misc]
    _LITEPARSE_AVAILABLE = False


class LiteParseLoader(DataLoader):
    """Loads and parses documents using LiteParse (local, no cloud dependency).

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
