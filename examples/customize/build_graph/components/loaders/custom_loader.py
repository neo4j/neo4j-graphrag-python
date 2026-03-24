"""Create a custom data loader to transform content into text."""

from pathlib import Path
from typing import Dict, Optional

from neo4j_graphrag.experimental.components.data_loader import DataLoader
from neo4j_graphrag.experimental.components.types import DocumentInfo, LoadedDocument


class MyLoader(DataLoader):
    async def run(
        self,
        filepath: Path,
        metadata: Optional[Dict[str, str]] = None,
    ) -> LoadedDocument:
        # Implement logic here
        return LoadedDocument(
            text="<extracted text>",
            document_info=DocumentInfo(
                path=str(filepath),
                metadata=metadata,
            ),
        )
