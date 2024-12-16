"""Use the PdfLoader component to extract text from a remote PDF file."""

import asyncio

from neo4j_graphrag.experimental.components.pdf_loader import PdfLoader

url = "https://raw.githubusercontent.com/neo4j/neo4j-graphrag-python/c166afc4d5abc56a5686f3da46a97ed7c07da19d/examples/data/Harry%20Potter%20and%20the%20Chamber%20of%20Secrets%20Summary.pdf"


async def main() -> None:
    loader = PdfLoader()
    document = await loader.run(filepath=url, fs="http")
    print(document.text[:100])


if __name__ == "__main__":
    asyncio.run(main())
