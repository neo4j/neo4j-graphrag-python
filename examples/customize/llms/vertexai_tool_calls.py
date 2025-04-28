"""
Example showing how to use VertexAI tool calls with parameter extraction.
Both synchronous and asynchronous examples are provided.
"""

import asyncio

from dotenv import load_dotenv
from vertexai.generative_models import GenerationConfig

from neo4j_graphrag.llm import VertexAILLM
from neo4j_graphrag.llm.types import ToolCallResponse
from neo4j_graphrag.tool import Tool, ObjectParameter, StringParameter, IntegerParameter

# Load environment variables from .env file
load_dotenv()


# Create a custom Tool implementation for person info extraction
parameters = ObjectParameter(
    description="Parameters for extracting person information",
    properties={
        "name": StringParameter(description="The person's full name"),
        "age": IntegerParameter(description="The person's age"),
        "occupation": StringParameter(description="The person's occupation"),
    },
    required_properties=["name"],
    additional_properties=False,
)


def run_tool(name: str, age: int, occupation: str) -> str:
    """A simple function that summarizes person information from input parameters."""
    return f"Found person {name} with age {age} and occupation {occupation}"


person_info_tool = Tool(
    name="extract_person_info",
    description="Extract information about a person from text",
    parameters=parameters,
    execute_func=run_tool,
)

# Create the tool instance
TOOLS = [person_info_tool]


def process_tool_call(response: ToolCallResponse) -> str:
    """Process the tool call response and return the extracted parameters."""
    if not response.tool_calls:
        raise ValueError("No tool calls found in response")

    tool_call = response.tool_calls[0]
    print(f"\nTool called: {tool_call.name}")
    print(f"Arguments: {tool_call.arguments}")
    print(f"Additional content: {response.content or 'None'}")
    return person_info_tool.execute(**tool_call.arguments)  # type: ignore[no-any-return]


async def main() -> None:
    # Initialize the VertexAI LLM
    generation_config = GenerationConfig(temperature=0.0)
    llm = VertexAILLM(
        model_name="gemini-1.5-flash-001",
        generation_config=generation_config,
    )

    # Example text containing information about a person
    text = "Stella Hane is a 35-year-old software engineer who loves coding."

    print("\n=== Synchronous Tool Call ===")
    # Make a synchronous tool call
    sync_response = llm.invoke_with_tools(
        input=f"Extract information about the person from this text: {text}",
        tools=TOOLS,
    )
    sync_result = process_tool_call(sync_response)
    print("\n=== Synchronous Tool Call Result ===")
    print(sync_result)

    print("\n=== Asynchronous Tool Call ===")
    # Make an asynchronous tool call with a different text
    text2 = "Molly Hane, 32, works as a data scientist and enjoys machine learning."
    async_response = await llm.ainvoke_with_tools(
        input=f"Extract information about the person from this text: {text2}",
        tools=TOOLS,
    )
    async_result = process_tool_call(async_response)
    print("\n=== Asynchronous Tool Call Result ===")
    print(async_result)


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
