#  Copyright (c) "Neo4j"
#  Neo4j Sweden AB [https://neo4j.com]
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#      https://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""
Simple example comparing OpenAI LLM V1 (legacy) vs V2 (structured output).

This demonstrates how V2's structured output provides type-safe, validated responses
compared to V1's prompt-based JSON extraction.

Prerequisites:
- OpenAI API key set in OPENAI_API_KEY environment variable
"""

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.types import LLMMessage

load_dotenv()

# Define a Pydantic model for structured output
class Movie(BaseModel):
    model_config = ConfigDict(extra="forbid") # This is important to prevent extra properties from being added to the response
    
    title: str
    year: int
    director: str
    genre: str


# =============================================================================
# V1 (Legacy): Manual JSON mode with prompt engineering
# =============================================================================
print("=" * 60)
print("V1 Legacy: Manual JSON extraction with prompt engineering")
print("=" * 60)

# V1: Use model_params with response_format for JSON object mode
llm_v1 = OpenAILLM(
    model_name="gpt-4o-mini",
    model_params={
        "response_format": {"type": "json_object"},
        "temperature": 0
    }
)

# V1 requires string input and explicit JSON instructions in the prompt
v1_prompt = """Extract movie information and respond in JSON format.
Include: title, year, director, genre.

Text: Inception was directed by Christopher Nolan in 2010. It's a science fiction thriller."""

response_v1 = llm_v1.invoke(v1_prompt)
print(f"Response: {response_v1.content}")


# =============================================================================
# V2 (New): Structured output with Pydantic model
# =============================================================================
print("\n" + "=" * 60)
print("V2: Structured output with Pydantic model")
print("=" * 60)

# V2: Use clean LLM without constructor params
llm_v2 = OpenAILLM(model_name="gpt-4o-mini")

# V2 uses list of LLMMessage for input
messages = [
    LLMMessage(
        role="user",
        content="Inception was directed by Christopher Nolan in 2010. It's a science fiction thriller."
    )
]

# Pass response_format and temperature directly to invoke()
response_v2 = llm_v2.invoke(
    messages,
    response_format=Movie,
    temperature=0
)

# Parse and validate in one step
movie = Movie.model_validate_json(response_v2.content)
print(f"Response: {response_v2.content}")


# =============================================================================
# V2: Using JSON Schema instead of Pydantic
# =============================================================================
print("\n" + "=" * 60)
print("V2 Alternative: Structured output with JSON Schema")
print("=" * 60)

# V2: Use clean LLM without constructor params
llm_v2 = OpenAILLM(model_name="gpt-4o-mini")

# V2 uses list of LLMMessage for input
messages = [
    LLMMessage(
        role="user",
        content="Inception was directed by Christopher Nolan in 2010. It's a science fiction thriller."
    )
]

# Define a JSON schema (equivalent to the Movie Pydantic model)
# Note: OpenAI requires JSON schemas to be wrapped in this specific format
movie_schema = {
    "type": "json_schema",
    "json_schema": {
        "name": "movie_info",
        "schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "year": {"type": "integer"},
                "director": {"type": "string"},
                "genre": {"type": "string"}
            },
            "required": ["title", "year", "director", "genre"],
            "additionalProperties": False
        }
    }
}

# Pass JSON schema as response_format
response_v2_schema = llm_v2.invoke(
    messages,
    response_format=movie_schema,
    temperature=0
)

print(f"Response: {response_v2_schema.content}")