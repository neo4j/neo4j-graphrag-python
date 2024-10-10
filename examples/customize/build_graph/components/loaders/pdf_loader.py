"""Use the PdfLoader component to extract text from a PDF file."""

import asyncio
from pathlib import Path

from neo4j_graphrag.experimental.components.pdf_loader import PdfLoader


async def main():
    loader = PdfLoader()
    document = await loader.run(
        filepath=Path(
            "../../../../data/Harry Potter and the Chamber of Secrets Summary.pdf"
        )
    )
    print(document.text[:30])
    print(document.document_info)


if __name__ == "__main__":
    asyncio.run(main())
