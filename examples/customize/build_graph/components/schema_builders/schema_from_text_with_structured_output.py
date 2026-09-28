"""
Simple example demonstrating structured output with SchemaFromTextExtractor.

This example shows how to use structured output for more reliable schema extraction
with automatic validation against the GraphSchema Pydantic model.

The GraphSchema is now compatible with both OpenAI and VertexAI structured output APIs,
with strict validation and proper field definitions. With structured output enabled:
- Uses structured output (list of messages)
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

from neo4j_graphrag.components.schema import (
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
                f"  - {constraint.type} on {constraint.node_type}.{list(constraint.property_names)}"
            )


async def test_prompt_based_extraction() -> GraphSchema:
    """
    Test the prompt-based approach (default): JSON extraction with manual cleanup.

    With use_structured_output=False (default):
    - Uses plain prompting (single user message)
    - LLM returns JSON string that needs parsing and cleanup
    - Extensive filtering and validation applied manually
    - More forgiving of LLM errors
    - Works with all LLM providers
    """
    print("\n" + "=" * 60)
    print("Testing prompt-based JSON extraction (default)")
    print("=" * 60)

    # Initialize LLM with response_format for JSON mode
    # gpt-4.1-mini rather than gpt-5-mini: this example pins temperature=0 so the
    # extracted schema is reproducible, and gpt-5 models only accept the default
    # temperature (1).
    llm = OpenAILLM(
        model_name="gpt-4.1-mini",
        model_params={
            "temperature": 0,
            "response_format": {"type": "json_object"},
        },
    )

    # For VertexAI, use:
    # llm = VertexAILLM(
    #     model_name="gemini-2.5-flash",
    #     model_params={"temperature": 0}
    # )

    # Create extractor WITHOUT structured output (the default)
    extractor = SchemaFromTextExtractor(
        llm=llm,
        use_structured_output=False,  # Default, can be omitted
    )

    # Extract schema
    schema = await extractor.run(text=SAMPLE_TEXT)

    print_schema_summary(schema, "Prompt-based Result")

    return schema


async def test_structured_output() -> GraphSchema:
    """
    Test the structured output approach with GraphSchema validation.

    With use_structured_output=True:
    - Uses structured output (list of messages)
    - Passes GraphSchema as response_format to ainvoke()
    - LLM returns properly structured data conforming to GraphSchema
    - Automatic validation via Pydantic
    - Less manual cleanup needed
    - Only works with OpenAI and VertexAI
    - Enforces min_length=1 on node properties
    """
    print("\n" + "=" * 60)
    print("Testing structured output with GraphSchema")
    print("=" * 60)

    # Initialize LLM - NO response_format in constructor for structured output!
    # gpt-4.1-mini for the same reason as above: temperature=0 is unsupported
    # on gpt-5 models.
    llm = OpenAILLM(model_name="gpt-4.1-mini", model_params={"temperature": 0})

    # For VertexAI, use:
    # llm = VertexAILLM(
    #     model_name="gemini-2.5-flash",
    #     model_params={"temperature": 0}
    # )

    # Create extractor WITH structured output
    extractor = SchemaFromTextExtractor(
        llm=llm,
        use_structured_output=True,  # This is the key parameter!
    )

    # Extract schema
    schema = await extractor.run(text=SAMPLE_TEXT)

    print_schema_summary(schema, "Structured Output Result")

    return schema


async def compare_approaches() -> None:
    """Run both approaches and compare results."""
    load_dotenv()

    # Test prompt-based extraction (default)
    schema_prompt_based = await test_prompt_based_extraction()

    # Test structured output
    schema_structured = await test_structured_output()

    # Comparison
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print("Prompt-based:")
    print(f"  - Node types: {len(schema_prompt_based.node_types)}")
    print(f"  - Relationship types: {len(schema_prompt_based.relationship_types)}")
    print(f"  - Patterns: {len(schema_prompt_based.patterns)}")
    print(
        f"  - Total properties: {sum(len(n.properties) for n in schema_prompt_based.node_types)}"
    )

    print("\nStructured Output:")
    print(f"  - Node types: {len(schema_structured.node_types)}")
    print(f"  - Relationship types: {len(schema_structured.relationship_types)}")
    print(f"  - Patterns: {len(schema_structured.patterns)}")
    print(
        f"  - Total properties: {sum(len(n.properties) for n in schema_structured.node_types)}"
    )


if __name__ == "__main__":
    # Run comparison between prompt-based and structured output extraction
    asyncio.run(compare_approaches())
