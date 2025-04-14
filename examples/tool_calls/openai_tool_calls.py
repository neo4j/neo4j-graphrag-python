"""
Example showing how to use OpenAI tool calls with parameter extraction.
Both synchronous and asynchronous examples are provided.

To run this example:
1. Make sure you have the OpenAI API key in your .env file:
   OPENAI_API_KEY=your-api-key
2. Run: python examples/tool_calls/openai_tool_calls.py
"""

import asyncio
import json
import os
from typing import Dict, Any

from dotenv import load_dotenv

from neo4j_graphrag.llm import OpenAILLM
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
person_info_tool = Tool(
    name="extract_person_info",
    description="Extract information about a person from text",
    parameters=parameters,
    execute_func=lambda **kwargs: kwargs,
)

# Create the tool instance
TOOLS = [person_info_tool]


def process_tool_call(response: ToolCallResponse) -> Dict[str, Any]:
    """Process the tool call response and return the extracted parameters."""
    if not response.tool_calls:
        raise ValueError("No tool calls found in response")

    tool_call = response.tool_calls[0]
    print(f"\nTool called: {tool_call.name}")
    print(f"Arguments: {tool_call.arguments}")
    print(f"Additional content: {response.content or 'None'}")
    return tool_call.arguments


async def main() -> None:
    # Initialize the OpenAI LLM
    llm = OpenAILLM(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="gpt-4o",
        model_params={"temperature": 0},
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
    print(json.dumps(sync_result, indent=2))

    print("\n=== Asynchronous Tool Call ===")
    # Make an asynchronous tool call with a different text
    text2 = "Molly Hane, 32, works as a data scientist and enjoys machine learning."
    async_response = await llm.ainvoke_with_tools(
        input=f"Extract information about the person from this text: {text2}",
        tools=TOOLS,
    )
    async_result = process_tool_call(async_response)
    print("\n=== Asynchronous Tool Call Result ===")
    print(json.dumps(async_result, indent=2))


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
