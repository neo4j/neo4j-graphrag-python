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

from typing import Any, Optional

from neo4j_graphrag.embeddings.base import Embedder
from neo4j_graphrag.exceptions import EmbeddingsGenerationError
from neo4j_graphrag.utils.rate_limit import RateLimitHandler, rate_limit_handler

try:
    import cohere
except ImportError:
    cohere = None  # type: ignore


class CohereEmbeddings(Embedder):
    def __init__(
        self,
        model: str = "",
        rate_limit_handler: Optional[RateLimitHandler] = None,
        *,
        input_type: str = "search_document",
        **kwargs: Any,
    ) -> None:
        """
        Args:
            model (str): The name of the Cohere embedding model to use.
            rate_limit_handler (Optional[RateLimitHandler]): Handler for rate limiting.
            input_type (str): How Cohere should interpret the text. Current Cohere
                models reject a request that omits it. Keyword-only, so that the
                positional order of the arguments above is unchanged.

                Cohere's embeddings are asymmetric, so this affects retrieval
                quality: ``search_document`` for text being indexed,
                ``search_query`` for a search query. It defaults to
                ``search_document``, which is what ``TextChunkEmbedder`` needs and
                the expensive side to get wrong - a mismatch while indexing means
                re-embedding the corpus.

                ``Embedder.embed_query()`` takes no per-call arguments, so
                retrievers cannot override this. Give the retrieval side its own
                instance::

                    indexer = CohereEmbeddings(model="embed-english-v3.0")
                    retriever_embedder = CohereEmbeddings(
                        model="embed-english-v3.0", input_type="search_query"
                    )

            kwargs: All other parameters are passed to the Cohere client.
        """
        if cohere is None:
            raise ImportError(
                """Could not import cohere python client.
                Please install it with `pip install "neo4j-graphrag[cohere]"`."""
            )
        super().__init__(rate_limit_handler)
        self.model = model
        self.input_type = input_type
        self.client = cohere.Client(**kwargs)

    @rate_limit_handler
    def embed_query(self, text: str, **kwargs: Any) -> list[float]:
        try:
            # setdefault so an explicit per-call input_type still wins, and so
            # passing one does not collide with the constructor's.
            kwargs.setdefault("input_type", self.input_type)
            response = self.client.embed(
                texts=[text],
                model=self.model,
                **kwargs,
            )
            return response.embeddings[0]  # type: ignore
        except Exception as e:
            raise EmbeddingsGenerationError(
                f"Failed to generate embedding with Cohere: {e}"
            ) from e
