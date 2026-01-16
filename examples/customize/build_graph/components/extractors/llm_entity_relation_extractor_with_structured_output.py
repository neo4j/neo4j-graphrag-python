"""
Simple example demonstrating structured output with LLMEntityRelationExtractor.

This example shows how to use structured output for more reliable entity and
relationship extraction with automatic schema validation.

The Neo4jGraph schema is now compatible with both OpenAI and VertexAI structured
output APIs, with strict schema validation (additionalProperties: false) and
proper required field definitions.

Prerequisites:
- Google Cloud credentials configured for VertexAI
- Or OpenAI API key set in OPENAI_API_KEY environment variable
"""

import asyncio
from dotenv import load_dotenv

from neo4j_graphrag.experimental.components.entity_relation_extractor import (
    LLMEntityRelationExtractor,
)
from neo4j_graphrag.experimental.components.types import (
    Neo4jGraph,
    TextChunk,
    TextChunks,
)
from neo4j_graphrag.llm import OpenAILLM, VertexAILLM


async def main() -> Neo4jGraph:
    """
    Demonstrates entity and relation extraction with structured output.

    With use_structured_output=True:
    - Uses LLMInterfaceV2 (list of messages)
    - Passes Neo4jGraph Pydantic model as response_format to invoke()
    - Ensures response conforms to expected graph structure
    - Provides automatic type validation
    - Reduces need for JSON repair and error handling
    """
    load_dotenv()
    # Initialize LLM - no response_format in constructor!
    llm = VertexAILLM(model_name="gemini-2.5-flash")

    # llm = OpenAILLM(
    #     model_name="gpt-4o-mini",
    #     model_params={"temperature": 0}
    # )

    # Enable structured output for reliable extraction
    extractor = LLMEntityRelationExtractor(
        llm=llm,
        use_structured_output=True,  # This is the key parameter!
    )

    # Sample text about a person and organization
    sample_text = """
    Albert Einstein was a theoretical physicist who developed the theory of relativity.
    He worked at the Institute for Advanced Study in Princeton from 1933 until his death in 1955.
    """

    # Extract entities and relationships
    graph = await extractor.run(
        chunks=TextChunks(chunks=[TextChunk(text=sample_text, index=0)])
    )

    return graph


if __name__ == "__main__":
    # Run extraction
    graph = asyncio.run(main())

    print(graph)
