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
import asyncio
from typing import TYPE_CHECKING, Any, Optional

from neo4j_graphrag.embeddings.base import Embedder, _split_into_groups
from neo4j_graphrag.exceptions import EmbeddingsGenerationError
from neo4j_graphrag.utils.rate_limit import RateLimitHandler, rate_limit_handler

if TYPE_CHECKING:
    import openai


class BaseOpenAIEmbeddings(Embedder, abc.ABC):
    """
    Abstract base class for OpenAI embeddings.
    """

    client: openai.OpenAI

    #: Maximum number of inputs sent in one embeddings request. The OpenAI
    #: embeddings endpoint accepts at most 2,048 inputs per request.
    max_inputs_per_request: int = 2048

    def __init__(
        self,
        model: str = "text-embedding-ada-002",
        rate_limit_handler: Optional[RateLimitHandler] = None,
        **kwargs: Any,
    ) -> None:
        try:
            import openai
        except ImportError:
            raise ImportError(
                """Could not import openai python client.
                Please install it with `pip install "neo4j-graphrag[openai]"`."""
            )
        super().__init__(rate_limit_handler)
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

    @rate_limit_handler
    def embed_query(self, text: str, **kwargs: Any) -> list[float]:
        """
        Generate embeddings for a given query using an OpenAI text embedding model.

        Args:
            text (str): The text to generate an embedding for.
            **kwargs (Any): Additional arguments to pass to the OpenAI embedding generation function.
        """
        try:
            response = self.client.embeddings.create(
                input=text, model=self.model, **kwargs
            )
            embedding: list[float] = response.data[0].embedding
            return embedding
        except Exception as e:
            raise EmbeddingsGenerationError(
                f"Failed to generate embedding with OpenAI: {e}"
            ) from e

    def embed_documents(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        """
        Generate embeddings for several texts, sending them to OpenAI in groups
        of at most ``max_inputs_per_request`` inputs per request instead of one
        request per text.

        Args:
            texts (list[str]): The texts to generate embeddings for.
            **kwargs (Any): Additional arguments to pass to the OpenAI embedding generation function.

        Returns:
            list[list[float]]: One embedding per text, in input order.
        """
        embeddings: list[list[float]] = []
        for group in _split_into_groups(texts, self.max_inputs_per_request):
            embeddings.extend(self._embed_group(group, **kwargs))
        return embeddings

    async def async_embed_documents(
        self, texts: list[str], **kwargs: Any
    ) -> list[list[float]]:
        """
        Asynchronous version of :meth:`embed_documents`. The blocking client
        call runs in a worker thread so that concurrent groups (for example in
        ``TextChunkEmbedder``) do not block the event loop.
        """
        return await asyncio.to_thread(self.embed_documents, texts, **kwargs)

    @rate_limit_handler
    def _embed_group(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        try:
            response = self.client.embeddings.create(
                input=texts, model=self.model, **kwargs
            )
        except Exception as e:
            raise EmbeddingsGenerationError(
                f"Failed to generate embeddings with OpenAI: {e}"
            ) from e
        # OpenAI tags every embedding with the position of its input; sort on it
        # rather than relying on the response order.
        data = sorted(response.data, key=lambda item: item.index)
        if len(data) != len(texts):
            raise EmbeddingsGenerationError(
                f"OpenAI returned {len(data)} embeddings for {len(texts)} inputs."
            )
        return [list(item.embedding) for item in data]

    @property
    def embedding_model_name(self) -> str:
        return self.model


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
