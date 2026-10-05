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
Simple example demonstrating OpenAI LLM structured output.

Structured output provides type-safe, validated responses.

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
    model_config = ConfigDict(
        extra="forbid"
    )  # This is important to prevent extra properties from being added to the response

    title: str
    year: int
    director: str
    genre: str


# gpt-4.1-mini rather than gpt-5-mini: this example pins temperature=0 so the
# extracted fields are reproducible, and gpt-5 models only accept the default
# temperature (1).
print("=" * 60)
print("Structured output with Pydantic model")
print("=" * 60)

with OpenAILLM(model_name="gpt-4.1-mini") as llm:
    messages = [
        LLMMessage(
            role="user",
            content="Inception was directed by Christopher Nolan in 2010. It's a science fiction thriller.",
        )
    ]

    # Pass response_format and temperature directly to invoke()
    response = llm.invoke(messages, response_format=Movie, temperature=0)

    # Parse and validate in one step
    movie = Movie.model_validate_json(response.content)
    print(f"Response: {response.content}")

    # =========================================================================
    # Alternative: Using JSON Schema instead of Pydantic
    # =========================================================================
    print("\n" + "=" * 60)
    print("Structured output with JSON Schema")
    print("=" * 60)

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
                    "genre": {"type": "string"},
                },
                "required": ["title", "year", "director", "genre"],
                "additionalProperties": False,
            },
        },
    }

    # Pass JSON schema as response_format
    response_schema = llm.invoke(messages, response_format=movie_schema, temperature=0)

    print(f"Response: {response_schema.content}")
