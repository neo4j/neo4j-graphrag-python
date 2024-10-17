"""Create a custom data loader to transform content into text."""

from pathlib import Path
from typing import Dict, Optional

from neo4j_graphrag.experimental.components.pdf_loader import (
    DataLoader,
    DocumentInfo,
    PdfDocument,
)
from pydantic import validate_call


class MyLoader(DataLoader):
    async def run(
        self, filepath: Path, metadata: Optional[Dict[str, str]] = None
    ) -> PdfDocument:
        # Implement logic here
        return PdfDocument(
            text="<extracted text>",
            document_info=DocumentInfo(
                path=str(filepath),
                # optionally, add some metadata as a dict
                metadata=None,
            ),
        )
