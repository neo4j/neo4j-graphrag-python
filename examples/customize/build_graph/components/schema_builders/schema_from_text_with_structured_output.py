"""
Simple example demonstrating structured output with SchemaFromTextExtractor.

This example shows how to use structured output for more reliable schema extraction
with automatic validation against the GraphSchema Pydantic model.

The GraphSchema is now compatible with both OpenAI and VertexAI structured output APIs,
with strict validation and proper field definitions. With structured output enabled:
- Uses LLMInterfaceV2 (list of messages)
- Passes GraphSchema Pydantic model as response_format to ainvoke()
- Ensures response conforms to expected schema structure
- Provides automatic type validation
- Reduces need for JSON repair and error handling
- Enforces min_length=1 on node properties (nodes must have at least one property)

Prerequisites:
- Google Cloud credentials configured for VertexAI
- Or OpenAI API key set in OPENAI_API_KEY environment variable
"""

import asyncio
from dotenv import load_dotenv

from neo4j_graphrag.experimental.components.schema import (
    SchemaFromTextExtractor,
    GraphSchema,
)
from neo4j_graphrag.llm import OpenAILLM


# Sample text to extract schema from
SAMPLE_TEXT = """
Acme Corporation was founded in 1985 by John Smith in New York City.
The company specializes in manufacturing high-quality widgets and gadgets
for the consumer electronics industry.

Sarah Johnson joined Acme in 2010 as a Senior Engineer and was promoted to
Engineering Director in 2015. She oversees a team of 12 engineers working on
next-generation products. Sarah holds a PhD in Electrical Engineering from MIT
and has filed 5 patents during her time at Acme.

The company expanded to international markets in 2012, opening offices in London,
Tokyo, and Berlin. Each office is managed by a regional director who reports
directly to the CEO, Michael Brown, who took over leadership in 2008.

Acme's most successful product, the SuperWidget X1, was launched in 2018 and
has sold over 2 million units worldwide. The product was developed by a team led
by Robert Chen, who joined the company in 2016 after working at TechGiant for 8 years.
"""


def print_schema_summary(schema: GraphSchema, title: str) -> None:
    """Print a formatted summary of the extracted schema."""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")

    print(f"\nNode Types ({len(schema.node_types)}):")
    for node in schema.node_types:
        props = [f"{p.name} ({p.type})" for p in node.properties]
        print(f"  - {node.label}")
        if props:
            print(f"    Properties: {', '.join(props)}")
        if node.description:
            print(f"    Description: {node.description}")

    if schema.relationship_types:
        print(f"\nRelationship Types ({len(schema.relationship_types)}):")
        for rel in schema.relationship_types:
            props = [f"{p.name} ({p.type})" for p in rel.properties]
            print(f"  - {rel.label}")
            if props:
                print(f"    Properties: {', '.join(props)}")

    if schema.patterns:
        print(f"\nPatterns ({len(schema.patterns)}):")
        for source, relationship, target in schema.patterns:
            print(f"  {source} --[{relationship}]--> {target}")

    if schema.constraints:
        print(f"\nConstraints ({len(schema.constraints)}):")
        for constraint in schema.constraints:
            print(
                f"  - {constraint.type} on {constraint.node_type}.{constraint.property_name}"
            )


async def test_v1_without_structured_output() -> GraphSchema:
    """
    Test V1 approach (default): Prompt-based JSON extraction with manual cleanup.

    With use_structured_output=False (default):
    - Uses LLMInterface V1 (plain string prompts)
    - LLM returns JSON string that needs parsing and cleanup
    - Extensive filtering and validation applied manually
    - More forgiving of LLM errors
    - Works with all LLM providers
    """
    print("\n" + "=" * 60)
    print("Testing V1: Prompt-based JSON extraction (default)")
    print("=" * 60)

    # Initialize LLM with response_format for JSON mode (V1 approach)
    llm = OpenAILLM(
        model_name="gpt-4o-mini",
        model_params={
            "temperature": 0,
            "response_format": {"type": "json_object"},  # JSON mode for V1
        },
    )

    # For VertexAI V1, use:
    # llm = VertexAILLM(
    #     model_name="gemini-2.5-flash",
    #     model_params={"temperature": 0}
    # )

    # Create extractor WITHOUT structured output (V1 default)
    extractor = SchemaFromTextExtractor(
        llm=llm,
        use_structured_output=False,  # Default, can be omitted
    )

    # Extract schema
    schema = await extractor.run(text=SAMPLE_TEXT)

    print_schema_summary(schema, "V1 Result (Prompt-based)")

    return schema


async def test_v2_with_structured_output() -> GraphSchema:
    """
    Test V2 approach: Structured output with GraphSchema validation.

    With use_structured_output=True:
    - Uses LLMInterfaceV2 (list of messages)
    - Passes GraphSchema as response_format to ainvoke()
    - LLM returns properly structured data conforming to GraphSchema
    - Automatic validation via Pydantic
    - Less manual cleanup needed
    - Only works with OpenAI and VertexAI
    - Enforces min_length=1 on node properties
    """
    print("\n" + "=" * 60)
    print("Testing V2: Structured output with GraphSchema")
    print("=" * 60)

    # Initialize LLM - NO response_format in constructor for V2!
    llm = OpenAILLM(model_name="gpt-4o-mini", model_params={"temperature": 0})

    # For VertexAI V2, use:
    # llm = VertexAILLM(
    #     model_name="gemini-2.5-flash",
    #     model_params={"temperature": 0}
    # )

    # Create extractor WITH structured output (V2)
    extractor = SchemaFromTextExtractor(
        llm=llm,
        use_structured_output=True,  # This is the key parameter!
    )

    # Extract schema
    schema = await extractor.run(text=SAMPLE_TEXT)

    print_schema_summary(schema, "V2 Result (Structured Output)")

    return schema


async def compare_approaches() -> None:
    """Run both approaches and compare results."""
    load_dotenv()

    # Test V1 (default)
    schema_v1 = await test_v1_without_structured_output()

    # Test V2 (structured output)
    schema_v2 = await test_v2_with_structured_output()

    # Comparison
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print("V1 (Prompt-based):")
    print(f"  - Node types: {len(schema_v1.node_types)}")
    print(f"  - Relationship types: {len(schema_v1.relationship_types)}")
    print(f"  - Patterns: {len(schema_v1.patterns)}")
    print(
        f"  - Total properties: {sum(len(n.properties) for n in schema_v1.node_types)}"
    )

    print("\nV2 (Structured Output):")
    print(f"  - Node types: {len(schema_v2.node_types)}")
    print(f"  - Relationship types: {len(schema_v2.relationship_types)}")
    print(f"  - Patterns: {len(schema_v2.patterns)}")
    print(
        f"  - Total properties: {sum(len(n.properties) for n in schema_v2.node_types)}"
    )


if __name__ == "__main__":
    # Run comparison between V1 and V2
    asyncio.run(compare_approaches())
