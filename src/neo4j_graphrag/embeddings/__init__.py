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
from .base import Embedder
from .cohere import CohereEmbeddings
from .mistral import MistralAIEmbeddings
from .openai import AzureOpenAIEmbeddings, OpenAIEmbeddings
from .sentence_transformers import SentenceTransformerEmbeddings
from .vertexai import VertexAIEmbeddings

__all__ = [
    "Embedder",
    "SentenceTransformerEmbeddings",
    "OpenAIEmbeddings",
    "AzureOpenAIEmbeddings",
    "VertexAIEmbeddings",
    "MistralAIEmbeddings",
    "CohereEmbeddings",
]
