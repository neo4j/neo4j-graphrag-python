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

from __future__ import annotations

from typing import Any, Type

from neo4j_graphrag.embeddings.base import Embedder

try:
    import openai
except ImportError:
    openai = None  # type: ignore


class OpenAIEmbeddings(Embedder):
    """
    OpenAI embeddings class.
    This class uses the OpenAI python client to generate embeddings for text data.

    Args:
        model (str): The name of the OpenAI embedding model to use. Defaults to "text-embedding-ada-002".
    """

    client_class: Type[openai.OpenAI] = openai.OpenAI

    def __init__(self, model: str = "text-embedding-ada-002", **kwargs: Any) -> None:
        if openai is None:
            raise ImportError(
                "Could not import openai python client. "
                "Please install it with `pip install openai`."
            )

        self.openai_model = self.client_class(**kwargs)
        self.model = model

    def embed_query(self, text: str, **kwargs: Any) -> list[float]:
        """
        Generate embeddings for a given query using a OpenAI text embedding model.

        Args:
            text (str): The text to generate an embedding for.
            **kwargs (Any): Additional arguments to pass to the OpenAI embedding generation function.
        """
        response = self.openai_model.embeddings.create(
            input=text, model=self.model, **kwargs
        )
        return response.data[0].embedding


class AzureOpenAIEmbeddings(OpenAIEmbeddings):
    client_class: Type[openai.OpenAI] = openai.AzureOpenAI
