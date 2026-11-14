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

import os
from typing import Any, Optional

from pydantic import BaseModel

from neo4j_graphrag.embeddings.base import Embedder
from neo4j_graphrag.exceptions import EmbeddingsGenerationError
from neo4j_graphrag.utils.rate_limit import RateLimitHandler, rate_limit_handler

try:
    import voyageai
except ImportError:
    voyageai = None  # type: ignore

_MATRYOSHKA_DIMENSIONS = (256, 512, 1024, 2048)
_FIXED_DIMENSIONS = (1024,)

# Models are listed explicitly: there is no default and no silent fallback.
# Maps the model name to the output dimensions it accepts (Voyage's default
# dimension is 1024 for all of them).
VOYAGE_EMBEDDING_MODELS: dict[str, tuple[int, ...]] = {
    "voyage-4-large": _MATRYOSHKA_DIMENSIONS,
    "voyage-4": _MATRYOSHKA_DIMENSIONS,
    "voyage-4-lite": _MATRYOSHKA_DIMENSIONS,
    "voyage-code-4": _MATRYOSHKA_DIMENSIONS,
    "voyage-3-large": _MATRYOSHKA_DIMENSIONS,
    "voyage-3.5": _MATRYOSHKA_DIMENSIONS,
    "voyage-3.5-lite": _MATRYOSHKA_DIMENSIONS,
    "voyage-code-3": _MATRYOSHKA_DIMENSIONS,
    "voyage-finance-2": _FIXED_DIMENSIONS,
    "voyage-law-2": _FIXED_DIMENSIONS,
}

VOYAGE_CONTEXTUALIZED_EMBEDDING_MODELS: dict[str, tuple[int, ...]] = {
    "voyage-context-4": _MATRYOSHKA_DIMENSIONS,
    "voyage-context-3": _MATRYOSHKA_DIMENSIONS,
}

# Combined token ceiling of one embeddings request; ``None`` where Voyage
# documents no limit (voyage-code-4), so only the input count applies.
VOYAGE_MAX_REQUEST_TOKENS: dict[str, Optional[int]] = {
    "voyage-4-large": 120_000,
    "voyage-4": 320_000,
    "voyage-4-lite": 1_000_000,
    "voyage-code-4": None,
    "voyage-3-large": 120_000,
    "voyage-3.5": 320_000,
    "voyage-3.5-lite": 1_000_000,
    "voyage-code-3": 120_000,
    "voyage-finance-2": 120_000,
    "voyage-law-2": 120_000,
}

# Combined token ceiling of a contextualized request with auto-chunking.
VOYAGE_CONTEXTUALIZED_MAX_REQUEST_TOKENS = 120_000

VOYAGE_INPUT_TYPES = ("document", "query")

# Voyage accepts at most 1000 inputs per embeddings request.
VOYAGE_MAX_INPUTS_PER_REQUEST = 1000

# Contextualized document embeddings are always requested with automatic
# chunking at the largest chunk size the API accepts.
CONTEXTUALIZED_CHUNK_SIZE = 32_000


def _validate_model(model: str, catalog: dict[str, tuple[int, ...]]) -> None:
    if not model:
        raise ValueError(
            f"A Voyage AI model is required, one of: {', '.join(catalog)}."
        )
    if model not in catalog:
        raise ValueError(
            f"Unsupported Voyage AI model {model!r} for this embedder, "
            f"expected one of: {', '.join(catalog)}."
        )


def _validate_input_type(input_type: str) -> None:
    if input_type not in VOYAGE_INPUT_TYPES:
        raise ValueError(
            f"input_type must be one of {VOYAGE_INPUT_TYPES}, got {input_type!r}."
        )


def _validate_output_dimension(
    model: str, output_dimension: Optional[int], catalog: dict[str, tuple[int, ...]]
) -> None:
    if output_dimension is None:
        return
    allowed = catalog[model]
    if output_dimension not in allowed:
        raise ValueError(
            f"{model} does not support output_dimension={output_dimension}, "
            f"expected one of: {', '.join(str(d) for d in allowed)}."
        )


def _error_detail(error: Exception) -> str:
    """Voyage error bodies are ``{"detail": "<text>"}``; the SDK exposes the text."""
    detail = getattr(error, "user_message", None) or str(error)
    if voyageai is not None and isinstance(error, voyageai.error.RateLimitError):
        # Lets the default rate limit handler recognise and retry the error.
        return f"Rate limit exceeded: {detail}"
    return str(detail)


def _count_tokens(client: Any, model: str, texts: list[str]) -> list[int]:
    """Token count of each text, from the tokenizer of the ``voyageai`` client.

    The count is computed locally, but the client downloads the tokenizer of the
    model from the Hugging Face Hub on first use (then it is cached), so this
    needs access to the Hub or a cached tokenizer. It is an estimate: with the
    ``document`` and ``query`` input types it came out at or above the usage
    the API reported.
    """
    try:
        return [len(encoding.ids) for encoding in client.tokenize(texts, model=model)]
    except Exception as e:
        raise EmbeddingsGenerationError(
            f"Failed to count tokens with VoyageAI: {_error_detail(e)}"
        ) from e


def _token_aware_batches(
    token_counts: list[int], max_tokens: Optional[int]
) -> list[range]:
    """Split inputs into consecutive batches within the input and token limits.

    A single input above ``max_tokens`` gets a batch of its own, where the API
    reports the error.
    """
    batches: list[range] = []
    start, tokens = 0, 0
    for i, count in enumerate(token_counts):
        full = i - start >= VOYAGE_MAX_INPUTS_PER_REQUEST or (
            max_tokens is not None and i > start and tokens + count > max_tokens
        )
        if full:
            batches.append(range(start, i))
            start, tokens = i, 0
        tokens += count
    batches.append(range(start, len(token_counts)))
    return batches


def _make_client(kwargs: dict[str, Any]) -> Any:
    if voyageai is None:
        raise ImportError(
            """Could not import voyageai python client.
            Please install it with `pip install "neo4j-graphrag[voyageai]"`."""
        )
    api_key = kwargs.pop("api_key", None)
    if api_key is None:
        api_key = os.getenv("VOYAGE_API_KEY")
    return voyageai.Client(api_key=api_key, **kwargs)  # type: ignore[attr-defined]


class VoyageAIEmbeddings(Embedder):
    """
    Voyage AI text embeddings, using the official ``voyageai`` client.

    There is no default model: it has to be one of
    :data:`VOYAGE_EMBEDDING_MODELS`. The API key is read from ``api_key`` or the
    ``VOYAGE_API_KEY`` environment variable.

    Args:
        model (str): The Voyage AI embedding model, for example ``voyage-4``.
        rate_limit_handler (Optional[RateLimitHandler]): Handler for rate limiting.
        input_type (str): ``document`` for text being indexed or ``query`` for a
            search query. Voyage's embeddings are asymmetric, and
            ``Embedder.embed_query()`` takes no per-call arguments, so give the
            retrieval side its own instance with ``input_type="query"``, as with
            :class:`~neo4j_graphrag.embeddings.cohere.CohereEmbeddings`. Defaults
            to ``document``, which is what ``TextChunkEmbedder`` needs.
        output_dimension (Optional[int]): Matryoshka output dimension (256, 512,
            1024 or 2048). Only sent when set; ``voyage-finance-2`` and
            ``voyage-law-2`` accept 1024 only.
        kwargs: All other parameters are passed to the Voyage AI client.
    """

    def __init__(
        self,
        model: str,
        rate_limit_handler: Optional[RateLimitHandler] = None,
        *,
        input_type: str = "document",
        output_dimension: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        _validate_model(model, VOYAGE_EMBEDDING_MODELS)
        _validate_input_type(input_type)
        _validate_output_dimension(model, output_dimension, VOYAGE_EMBEDDING_MODELS)
        super().__init__(rate_limit_handler)
        self.model = model
        self.input_type = input_type
        self.output_dimension = output_dimension
        self.client = _make_client(kwargs)

    @rate_limit_handler
    def _embed(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        kwargs.setdefault("input_type", self.input_type)
        _validate_input_type(kwargs["input_type"])
        if self.output_dimension is not None:
            kwargs.setdefault("output_dimension", self.output_dimension)
        try:
            response = self.client.embed(texts, model=self.model, **kwargs)
        except Exception as e:
            raise EmbeddingsGenerationError(
                f"Failed to generate embedding with VoyageAI: {_error_detail(e)}"
            ) from e
        embeddings = getattr(response, "embeddings", None)
        if not embeddings or len(embeddings) != len(texts):
            raise EmbeddingsGenerationError(
                "Failed to generate embedding with VoyageAI: expected "
                f"{len(texts)} embeddings, got {len(embeddings or [])}."
            )
        return embeddings  # type: ignore[no-any-return]

    def embed_query(self, text: str, **kwargs: Any) -> list[float]:
        """Embed a single text.

        Args:
            text (str): The text to embed.
            **kwargs (Any): Passed to the Voyage AI client; an explicit
                ``input_type`` wins over the constructor's.
        """
        return self._embed([text], **kwargs)[0]

    def embed_documents(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        """Embed many texts, in as many requests as the limits of the model need.

        The texts are split into consecutive batches of at most 1000 inputs
        and at most the combined token limit of the model per request
        (:data:`VOYAGE_MAX_REQUEST_TOKENS`), counted with the tokenizer of the
        ``voyageai`` client. The vectors come back in the order of ``texts``.
        """
        if not texts:
            raise ValueError("texts must not be empty.")
        max_tokens = VOYAGE_MAX_REQUEST_TOKENS[self.model]
        # Without a documented token ceiling only the input count limits a batch.
        token_counts = (
            [0] * len(texts)
            if max_tokens is None
            else _count_tokens(self.client, self.model, texts)
        )
        embeddings: list[list[float]] = []
        for batch in _token_aware_batches(token_counts, max_tokens):
            embeddings.extend(self._embed(texts[batch.start : batch.stop], **kwargs))
        return embeddings

    @property
    def embedding_model_name(self) -> str:
        return self.model


class ContextualizedDocumentEmbeddings(BaseModel):
    """Embeddings of one document: one vector per chunk, in chunk order.

    ``chunks`` holds the text of each chunk as returned by the API.
    """

    embeddings: list[list[float]]
    chunks: list[str]


class VoyageAIContextualizedEmbeddings(Embedder):
    """
    Voyage AI contextualized chunk embeddings (``voyage-context-4`` and
    ``voyage-context-3``): each chunk is embedded with the context of its whole
    document.

    Documents are embedded with :meth:`embed_documents`. It always asks the API
    to chunk automatically (``enable_auto_chunking=True``, ``chunk_size=32000``),
    which requires one string per document and ``input_type="document"``.
    :meth:`embed_query` embeds a search query and never chunks. Because it uses
    the query side, this class is the retrieval counterpart of the index and is
    not meant for ``TextChunkEmbedder``, which embeds chunks with ``embed_query``.

    There is no default model. The API key is read from ``api_key`` or the
    ``VOYAGE_API_KEY`` environment variable.

    Args:
        model (str): ``voyage-context-4`` or ``voyage-context-3``.
        rate_limit_handler (Optional[RateLimitHandler]): Handler for rate limiting.
        output_dimension (Optional[int]): 256, 512, 1024 or 2048. Only sent when set.
        kwargs: All other parameters are passed to the Voyage AI client.
    """

    def __init__(
        self,
        model: str,
        rate_limit_handler: Optional[RateLimitHandler] = None,
        *,
        output_dimension: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        _validate_model(model, VOYAGE_CONTEXTUALIZED_EMBEDDING_MODELS)
        _validate_output_dimension(
            model, output_dimension, VOYAGE_CONTEXTUALIZED_EMBEDDING_MODELS
        )
        super().__init__(rate_limit_handler)
        self.model = model
        self.output_dimension = output_dimension
        self.client = _make_client(kwargs)

    @rate_limit_handler
    def _contextualized_embed(self, inputs: list[Any], **kwargs: Any) -> Any:
        if self.output_dimension is not None:
            kwargs.setdefault("output_dimension", self.output_dimension)
        try:
            return self.client.contextualized_embed(
                inputs=inputs, model=self.model, **kwargs
            )
        except Exception as e:
            raise EmbeddingsGenerationError(
                f"Failed to generate embedding with VoyageAI: {_error_detail(e)}"
            ) from e

    def embed_query(self, text: str, **kwargs: Any) -> list[float]:
        """Embed a search query (``input_type="query"``, no chunking)."""
        if kwargs.pop("input_type", "query") != "query":
            raise ValueError("embed_query only supports input_type='query'.")
        response = self._contextualized_embed([[text]], input_type="query", **kwargs)
        results = getattr(response, "results", None)
        if not results or not results[0].embeddings:
            raise EmbeddingsGenerationError(
                "Failed to generate embedding with VoyageAI: no embedding returned."
            )
        return results[0].embeddings[0]  # type: ignore[no-any-return]

    def embed_documents(
        self, documents: list[str]
    ) -> list[ContextualizedDocumentEmbeddings]:
        """Chunk and embed whole documents, one string per document.

        The documents are sent in consecutive batches of at most 1000 documents
        and :data:`VOYAGE_CONTEXTUALIZED_MAX_REQUEST_TOKENS` combined tokens per
        request. Returns one :class:`ContextualizedDocumentEmbeddings` per
        document, in order, with the chunk texts the API produced.
        """
        if not documents:
            raise ValueError("documents must not be empty.")
        if not all(isinstance(d, str) for d in documents):
            raise ValueError(
                "Auto-chunking expects one string per document; pre-chunked "
                "lists are not supported."
            )
        token_counts = _count_tokens(self.client, self.model, documents)
        out: list[ContextualizedDocumentEmbeddings] = []
        for batch in _token_aware_batches(
            token_counts, VOYAGE_CONTEXTUALIZED_MAX_REQUEST_TOKENS
        ):
            out.extend(self._embed_document_batch(documents[batch.start : batch.stop]))
        return out

    def _embed_document_batch(
        self, documents: list[str]
    ) -> list[ContextualizedDocumentEmbeddings]:
        response = self._contextualized_embed(
            documents,
            input_type="document",
            enable_auto_chunking=True,
            chunk_size=CONTEXTUALIZED_CHUNK_SIZE,
        )
        results = getattr(response, "results", None)
        if not results or len(results) != len(documents):
            raise EmbeddingsGenerationError(
                "Failed to generate embedding with VoyageAI: expected "
                f"{len(documents)} documents, got {len(results or [])}."
            )
        out: list[ContextualizedDocumentEmbeddings] = []
        for result in sorted(results, key=lambda r: r.index):
            chunks = getattr(result, "chunk_texts", None)
            if not chunks or len(chunks) != len(result.embeddings):
                raise EmbeddingsGenerationError(
                    "Failed to generate embedding with VoyageAI: the response does "
                    "not return the text of every chunk."
                )
            out.append(
                ContextualizedDocumentEmbeddings(
                    embeddings=result.embeddings, chunks=chunks
                )
            )
        return out

    @property
    def embedding_model_name(self) -> str:
        return self.model
