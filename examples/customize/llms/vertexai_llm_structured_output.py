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
Simple example comparing VertexAI LLM V1 (legacy) vs V2 (structured output).

This demonstrates how V2's structured output provides type-safe, validated responses
compared to V1's prompt-based JSON extraction.

Prerequisites:
- Google Cloud project with Vertex AI API enabled
- Either:
  - GOOGLE_APPLICATION_CREDENTIALS environment variable set, or
  - Running on GCP with appropriate service account
"""

from pydantic import BaseModel
from neo4j_graphrag.llm import VertexAILLM
from neo4j_graphrag.types import LLMMessage
from vertexai.generative_models import GenerationConfig

# Define a Pydantic model for structured output
class Movie(BaseModel):
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

# V1: Use generation_config 
llm_v1 = VertexAILLM(
    model_name="gemini-2.5-flash",
    generation_config=GenerationConfig(
        response_mime_type="application/json",
        temperature=0
    )
)

# V1 requires string input
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
llm_v2 = VertexAILLM(model_name="gemini-2.5-flash")

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
# V2 Alternative: Using JSON Schema instead of Pydantic
# =============================================================================
print("\n" + "=" * 60)
print("V2 Alternative: Structured output with JSON Schema")
print("=" * 60)

# Define a JSON schema (equivalent to the Movie Pydantic model)
# Note: VertexAI accepts raw JSON schemas (no wrapping required like OpenAI)
movie_schema = {
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

# Pass JSON schema as response_format
response_v2_schema = llm_v2.invoke(
    messages,
    response_format=movie_schema,
    temperature=0
)

print(f"Response: {response_v2_schema.content}")
