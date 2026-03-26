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
"""Document loaders: base class, PDF, Markdown, and extension-based file dispatch."""

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import Dict, Optional, Union

import pypdf

from neo4j_graphrag.exceptions import MarkdownLoadError, PdfLoaderError
from neo4j_graphrag.experimental.components.types import (
    DocumentInfo,
    DocumentType,
    LoadedDocument,
)
from neo4j_graphrag.experimental.pipeline.component import Component


class DataLoader(Component):
    """
    Interface for loading data of various input types.
    """

    def get_document_metadata(
        self, text: str, metadata: Optional[Dict[str, str]] = None
    ) -> Dict[str, str] | None:
        return metadata

    @staticmethod
    def _apply_max_chars(text: str, max_chars: Optional[int] = None) -> str:
        if max_chars is None:
            return text
        if max_chars < 0:
            raise ValueError("max_chars must be >= 0")
        return text[:max_chars]

    @abstractmethod
    async def run(
        self,
        filepath: Union[str, Path],
        metadata: Optional[Dict[str, str]] = None,
        max_chars: Optional[int] = None,
    ) -> LoadedDocument: ...


class PdfLoader(DataLoader):
    """Loads text from PDF files using pypdf."""

    @staticmethod
    def load_file(file: str) -> str:
        """Parse a PDF file and return extracted text."""
        try:
            with open(file, "rb") as fp:
                pdf = pypdf.PdfReader(fp)
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
        max_chars: Optional[int] = None,
    ) -> LoadedDocument:
        if not isinstance(filepath, str):
            filepath = str(filepath)
        text = self._apply_max_chars(self.load_file(filepath), max_chars=max_chars)
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
    def load_file(file: str) -> str:
        try:
            with open(file, "rb") as fp:
                raw = fp.read()
            return raw.decode("utf-8")
        except Exception as e:
            raise MarkdownLoadError(e)

    async def run(
        self,
        filepath: Union[str, Path],
        metadata: Optional[Dict[str, str]] = None,
        max_chars: Optional[int] = None,
    ) -> LoadedDocument:
        if not isinstance(filepath, str):
            filepath = str(filepath)
        text = self._apply_max_chars(
            MarkdownLoader.load_file(filepath), max_chars=max_chars
        )
        return LoadedDocument(
            text=text,
            document_info=DocumentInfo(
                path=filepath,
                metadata=self.get_document_metadata(text, metadata),
                document_type=DocumentType.MARKDOWN,
            ),
        )
