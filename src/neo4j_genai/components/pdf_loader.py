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
import io
from abc import abstractmethod
from pathlib import Path
from typing import Optional

import fsspec
from fsspec import AbstractFileSystem
from fsspec.implementations.local import LocalFileSystem

from neo4j_genai.pipeline import Component, DataModel


class PdfDocument(DataModel):
    text: str


class DataLoader(Component):
    """
    Interface for loading data of various input types.
    """

    @abstractmethod
    async def run(self, filepath: str) -> PdfDocument:
        pass


def is_default_fs(fs: fsspec.AbstractFileSystem) -> bool:
    return isinstance(fs, LocalFileSystem) and not fs.auto_mkdir


class PdfLoader(DataLoader):
    def load_file(
        self,
        file: Path,
        extra_info: Optional[dict] = None,
        fs: Optional[AbstractFileSystem] = None,
    ) -> str:
        """Parse file."""
        if not isinstance(file, Path):
            file = Path(file)

        try:
            import pypdf
        except ImportError:
            raise ImportError(
                "pypdf is required to read PDF files: `pip install pypdf`"
            )
        fs = fs or LocalFileSystem()

        with fs.open(file, "rb") as fp:
            stream = fp if is_default_fs(fs) else io.BytesIO(fp.read())
            pdf = pypdf.PdfReader(stream)
            num_pages = len(pdf.pages)
            text = "\n".join(
                pdf.pages[page].extract_text() for page in range(num_pages)
            )

            return text

    async def run(self, filepath: Path) -> PdfDocument:
        return PdfDocument(text=self.load_file(filepath))
