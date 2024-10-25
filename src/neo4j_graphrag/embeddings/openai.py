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

import abc
from typing import TYPE_CHECKING, Any

from neo4j_graphrag.embeddings.base import Embedder

if TYPE_CHECKING:
    import openai


class BaseOpenAIEmbeddings(Embedder, abc.ABC):
    """
    Abstract base class for OpenAI embeddings.
    """

    client: openai.OpenAI

    def __init__(self, model: str = "text-embedding-ada-002", **kwargs: Any) -> None:
        try:
            import openai
        except ImportError:
            raise ImportError(
                "Could not import openai python client. "
                "Please install it with `pip install openai`."
            )
        self.openai = openai
        self.model = model
        self.client = self._initialize_client(**kwargs)

    @abc.abstractmethod
    def _initialize_client(self, **kwargs: Any) -> Any:
        """
        Initialize the OpenAI client.
        Must be implemented by subclasses.
        """
        pass

    def embed_query(self, text: str, **kwargs: Any) -> list[float]:
        """
        Generate embeddings for a given query using an OpenAI text embedding model.

        Args:
            text (str): The text to generate an embedding for.
            **kwargs (Any): Additional arguments to pass to the OpenAI embedding generation function.
        """
        response = self.client.embeddings.create(input=text, model=self.model, **kwargs)
        embedding: list[float] = response.data[0].embedding
        return embedding


class OpenAIEmbeddings(BaseOpenAIEmbeddings):
    """
    OpenAI embeddings class.
    This class uses the OpenAI python client to generate embeddings for text data.

    Args:
        model (str): The name of the OpenAI embedding model to use. Defaults to "text-embedding-ada-002".
        kwargs: All other parameters will be passed to the openai.OpenAI init.
    """

    def _initialize_client(self, **kwargs: Any) -> Any:
        return self.openai.OpenAI(**kwargs)


class AzureOpenAIEmbeddings(BaseOpenAIEmbeddings):
    """
    Azure OpenAI embeddings class.
    This class uses the Azure OpenAI python client to generate embeddings for text data.

    Args:
        model (str): The name of the Azure OpenAI embedding model to use. Defaults to "text-embedding-ada-002".
        kwargs: All other parameters will be passed to the openai.AzureOpenAI init.
    """

    def _initialize_client(self, **kwargs: Any) -> Any:
        return self.openai.AzureOpenAI(**kwargs)
