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
Simple example comparing Anthropic LLM V1 (legacy) vs V2 (structured output).

This demonstrates how V2's structured output provides type-safe, validated responses
compared to V1's prompt-based JSON extraction.

Structured output relies on Anthropic's native ``output_config`` API, which is
available on recent Claude models (Claude Sonnet 4.5 / Opus 4.5 and newer).

Prerequisites:
- Anthropic API key set in the ANTHROPIC_API_KEY environment variable
"""

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict
from neo4j_graphrag.llm import AnthropicLLM
from neo4j_graphrag.types import LLMMessage

load_dotenv()


# Define a Pydantic model for structured output
class Movie(BaseModel):
    model_config = ConfigDict(
        extra="forbid"
    )  # This is important to prevent extra properties from being added to the response

    title: str
    year: int
    director: str
    genre: str


# =============================================================================
# V1 (Legacy): Manual JSON extraction with prompt engineering
# =============================================================================
print("=" * 60)
print("V1 Legacy: Manual JSON extraction with prompt engineering")
print("=" * 60)

with (
    AnthropicLLM(
        model_name="claude-sonnet-4-5",
        model_params={"max_tokens": 1000},  # max_tokens must be specified
    ) as llm_v1,
    AnthropicLLM(
        model_name="claude-sonnet-4-5",
        model_params={"max_tokens": 1000},
    ) as llm_v2,
):
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

    # V2 uses list of LLMMessage for input
    messages = [
        LLMMessage(
            role="user",
            content="Inception was directed by Christopher Nolan in 2010. It's a science fiction thriller.",
        )
    ]

    # Pass response_format and temperature directly to invoke()
    response_v2 = llm_v2.invoke(messages, response_format=Movie, temperature=0)

    # Parse and validate in one step
    movie = Movie.model_validate_json(response_v2.content)
    print(f"Response: {response_v2.content}")

    # =============================================================================
    # V2 Alternative: Using a JSON Schema instead of Pydantic
    # =============================================================================
    print("\n" + "=" * 60)
    print("V2 Alternative: Structured output with JSON Schema")
    print("=" * 60)

    # When passing a dict, Anthropic expects it already shaped as an output_config,
    # i.e. wrapped in {"format": {"type": "json_schema", "schema": ...}}.
    movie_schema = {
        "format": {
            "type": "json_schema",
            "schema": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "year": {"type": "integer"},
                    "director": {"type": "string"},
                    "genre": {"type": "string"},
                },
                "required": ["title", "year", "director", "genre"],
                "additionalProperties": False,
            },
        }
    }

    # Pass JSON schema as response_format
    response_v2_schema = llm_v2.invoke(
        messages, response_format=movie_schema, temperature=0
    )

    print(f"Response: {response_v2_schema.content}")
