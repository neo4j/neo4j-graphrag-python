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
from neo4j_genai.experimental.components.pdf_loader import PdfLoader
from neo4j_genai.exceptions import PdfLoaderError

BASE_DIR = Path(__file__).parent


@pytest.fixture
def pdf_loader() -> PdfLoader:
    return PdfLoader()


@pytest.fixture
def dummy_pdf_path() -> Path:
    return BASE_DIR / "sample_data/lorem_ipsum.pdf"


def test_pdf_loading(pdf_loader: PdfLoader, dummy_pdf_path: Path) -> None:
    expected_content = "Lorem ipsum dolor sit amet."
    actual_content = pdf_loader.load_file(dummy_pdf_path)
    assert actual_content == expected_content


def test_pdf_processing_error(pdf_loader: PdfLoader, dummy_pdf_path: Path) -> None:
    with patch(
        "fsspec.implementations.local.LocalFileSystem.open",
        side_effect=Exception("Failed to open"),
    ):
        with pytest.raises(PdfLoaderError):
            pdf_loader.load_file(dummy_pdf_path)
