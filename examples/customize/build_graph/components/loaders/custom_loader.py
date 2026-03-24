"""Create a custom data loader to transform content into text."""

from pathlib import Path
from typing import Dict, Optional, Union

from fsspec import AbstractFileSystem

from neo4j_graphrag.experimental.components.data_loader import DataLoader
from neo4j_graphrag.experimental.components.types import DocumentInfo, LoadedDocument


class MyLoader(DataLoader):
    async def run(
        self,
        filepath: Union[str, Path],
        metadata: Optional[Dict[str, str]] = None,
        fs: Optional[Union[AbstractFileSystem, str]] = None,
    ) -> LoadedDocument:
        # Implement logic here; use ``fs`` when reading from non-local storage.
        _ = fs
        return LoadedDocument(
            text="<extracted text>",
            document_info=DocumentInfo(
                path=str(filepath),
                metadata=metadata,
            ),
        )
