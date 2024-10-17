"""Use the PdfLoader component to extract text from a PDF file."""

import asyncio
from pathlib import Path

from neo4j_graphrag.experimental.components.pdf_loader import PdfLoader

root_dir = Path(__file__).parents[4]
file_path = root_dir / "data" / "Harry Potter and the Chamber of Secrets Summary.pdf"


async def main() -> None:
    loader = PdfLoader()
    document = await loader.run(filepath=file_path)
    print(document.text[:200])
    print(document.document_info)


if __name__ == "__main__":
    asyncio.run(main())
