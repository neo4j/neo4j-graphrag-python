"""Create a custom data loader to transform content into text."""

from pathlib import Path
from typing import Dict, Optional

from neo4j_graphrag.experimental.components.pdf_loader import (
    DataLoader,
    DocumentInfo,
    PdfDocument,
)
from pydantic import BaseModel, validate_call


class InputModel(BaseModel):
    filepath: Path
    metadata: Optional[Dict[str, str]] = None


class MyLoader(DataLoader):
    @validate_call
    async def run(self, data: InputModel) -> PdfDocument:
        # Implement logic here
        return PdfDocument(
            text="<extracted text>",
            document_info=DocumentInfo(
                path=str(data.filepath),
                # optionally, add some metadata as a dict
                metadata=None,
            ),
        )
