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
from typing import TYPE_CHECKING, Any

# Same lazy-export rationale as ``neo4j_graphrag.llm``: importing the package
# must not eagerly import every provider SDK. ``VertexAIEmbeddings`` pulls in
# ``google.cloud.aiplatform`` (~1s+) and ``SentenceTransformerEmbeddings`` pulls
# in ``torch`` (hundreds of MB), so both are resolved on first access instead.
from .base import Embedder

_LAZY_EXPORTS: dict[str, str] = {
    "BedrockEmbeddings": ".bedrock",
    "CohereEmbeddings": ".cohere",
    "GeminiEmbedder": ".google_genai",
    "MistralAIEmbeddings": ".mistral",
    "OllamaEmbeddings": ".ollama",
    "AzureOpenAIEmbeddings": ".openai",
    "OpenAIEmbeddings": ".openai",
    "SentenceTransformerEmbeddings": ".sentence_transformers",
    "VertexAIEmbeddings": ".vertexai",
}

if TYPE_CHECKING:
    # Static-only mirror of _LAZY_EXPORTS so type checkers still resolve the
    # embedder classes; has no effect at runtime.
    from .bedrock import BedrockEmbeddings
    from .cohere import CohereEmbeddings
    from .google_genai import GeminiEmbedder
    from .mistral import MistralAIEmbeddings
    from .ollama import OllamaEmbeddings
    from .openai import AzureOpenAIEmbeddings, OpenAIEmbeddings
    from .sentence_transformers import SentenceTransformerEmbeddings
    from .vertexai import VertexAIEmbeddings

__all__ = [
    "Embedder",
    "BedrockEmbeddings",
    "SentenceTransformerEmbeddings",
    "OllamaEmbeddings",
    "OpenAIEmbeddings",
    "AzureOpenAIEmbeddings",
    "VertexAIEmbeddings",
    "MistralAIEmbeddings",
    "CohereEmbeddings",
    "GeminiEmbedder",
]


def __getattr__(name: str) -> Any:
    """Resolve lazily-exported embedder classes on first access (see module docstring)."""
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is not None:
        from importlib import import_module

        value = getattr(import_module(module_name, __package__), name)
        globals()[name] = value  # cache: future lookups bypass __getattr__
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
