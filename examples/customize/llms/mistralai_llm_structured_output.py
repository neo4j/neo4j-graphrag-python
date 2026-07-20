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
"""Use MistralAI structured output with a Pydantic model or JSON schema.

Prerequisites:
- MistralAI API key set in the MISTRAL_API_KEY environment variable
"""

from dotenv import load_dotenv
from neo4j_graphrag.llm import MistralAILLM
from neo4j_graphrag.types import LLMMessage
from pydantic import BaseModel

load_dotenv()


class Book(BaseModel):
    name: str
    authors: list[str]


messages = [
    LLMMessage(role="system", content="Extract the book information."),
    LLMMessage(
        role="user",
        content="I recently read 'To Kill a Mockingbird' by Harper Lee.",
    ),
]

llm = MistralAILLM(model_name="ministral-8b-latest")

# Pydantic models use Mistral's chat.parse API internally.
response = llm.invoke(messages, response_format=Book, temperature=0)
book = Book.model_validate_json(response.content)
print(book)

# Provider-specific dictionaries are passed to chat.complete unchanged.
book_response_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "Book",
        "schema": Book.model_json_schema(),
        "strict": True,
    },
}
response = llm.invoke(
    messages,
    response_format=book_response_format,
    temperature=0,
)
print(response.content)
