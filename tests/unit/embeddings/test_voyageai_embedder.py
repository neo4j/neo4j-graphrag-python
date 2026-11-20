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
from typing import Any, Iterator
from unittest.mock import MagicMock, Mock, patch

import pytest
from tenacity import RetryError
from voyageai.error import InvalidRequestError, RateLimitError

from neo4j_graphrag.embeddings.voyageai import (
    VoyageAIContextualizedEmbeddings,
    VoyageAIEmbeddings,
)
from neo4j_graphrag.exceptions import EmbeddingsGenerationError
from neo4j_graphrag.utils.rate_limit import RetryRateLimitHandler


@pytest.fixture
def client_cls() -> Iterator[Mock]:
    with patch("neo4j_graphrag.embeddings.voyageai.voyageai.Client") as cls:
        yield cls


@patch("neo4j_graphrag.embeddings.voyageai.voyageai", None)
def test_voyageai_missing_dependency() -> None:
    with pytest.raises(ImportError):
        VoyageAIEmbeddings(model="voyage-4")
    with pytest.raises(ImportError):
        VoyageAIContextualizedEmbeddings(model="voyage-context-4")


def test_model_is_required() -> None:
    with pytest.raises(TypeError):
        VoyageAIEmbeddings()  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        VoyageAIContextualizedEmbeddings()  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="model is required"):
        VoyageAIEmbeddings(model="")


@pytest.mark.parametrize(
    "model", ["not-a-model", "voyage-context-4", "rerank-3", "embed-english-v3.0"]
)
def test_unknown_or_foreign_model_is_rejected(model: str, client_cls: Mock) -> None:
    with pytest.raises(ValueError, match="Unsupported Voyage AI model"):
        VoyageAIEmbeddings(model=model)
    client_cls.assert_not_called()


def test_contextualized_rejects_text_embedding_model(client_cls: Mock) -> None:
    with pytest.raises(ValueError, match="Unsupported Voyage AI model"):
        VoyageAIContextualizedEmbeddings(model="voyage-4")


@pytest.mark.parametrize(
    "model, dimension",
    [("voyage-4", 768), ("voyage-law-2", 512), ("voyage-finance-2", 2048)],
)
def test_unsupported_output_dimension(
    model: str, dimension: int, client_cls: Mock
) -> None:
    with pytest.raises(ValueError, match="output_dimension"):
        VoyageAIEmbeddings(model=model, output_dimension=dimension)


def test_invalid_input_type(client_cls: Mock) -> None:
    with pytest.raises(ValueError, match="input_type"):
        VoyageAIEmbeddings(model="voyage-4", input_type="search_query")
    embedder = VoyageAIEmbeddings(model="voyage-4")
    with pytest.raises(ValueError, match="input_type"):
        embedder.embed_query("x", input_type="classification")
    client_cls.return_value.embed.assert_not_called()


def test_embed_query_request(client_cls: Mock) -> None:
    client_cls.return_value.embed.return_value = MagicMock(embeddings=[[1.0, 2.0]])
    embedder = VoyageAIEmbeddings(
        model="voyage-4", input_type="query", output_dimension=256, api_key="k"
    )
    assert embedder.embed_query("my text") == [1.0, 2.0]
    client_cls.assert_called_once_with(api_key="k")
    client_cls.return_value.embed.assert_called_once_with(
        ["my text"], model="voyage-4", input_type="query", output_dimension=256
    )


def test_output_dimension_only_sent_when_requested(client_cls: Mock) -> None:
    client_cls.return_value.embed.return_value = MagicMock(embeddings=[[1.0]])
    VoyageAIEmbeddings(model="voyage-4").embed_query("x")
    _, kwargs = client_cls.return_value.embed.call_args
    assert "output_dimension" not in kwargs
    assert kwargs["input_type"] == "document"


def test_api_key_comes_from_environment(
    client_cls: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VOYAGE_API_KEY", "from-env")
    VoyageAIEmbeddings(model="voyage-4")
    client_cls.assert_called_once_with(api_key="from-env")


def _tokens(*counts: int) -> list[Any]:
    return [MagicMock(ids=[0] * c) for c in counts]


def _embed_by_text(texts: list[str], **_: Any) -> Any:
    return MagicMock(embeddings=[[float(t)] for t in texts])


def test_embed_documents_must_not_be_empty(client_cls: Mock) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        VoyageAIEmbeddings(model="voyage-4").embed_documents([])
    client_cls.return_value.embed.assert_not_called()


@pytest.mark.parametrize(
    "model, counts, expected_batches",
    [
        # 320K per request for voyage-4
        ("voyage-4", [200_000, 200_000, 100_000], [["0"], ["1", "2"]]),
        # 120K for voyage-4-large, 1M for the lite models
        ("voyage-4-large", [70_000, 70_000, 40_000], [["0"], ["1", "2"]]),
        ("voyage-4-lite", [600_000, 600_000], [["0"], ["1"]]),
        # no documented ceiling for voyage-code-4
        ("voyage-code-4", [900_000, 900_000], [["0", "1"]]),
        # reaching the ceiling exactly still fits; an oversized input is alone
        ("voyage-4", [160_000, 160_000], [["0", "1"]]),
        ("voyage-4", [400_000, 1], [["0"], ["1"]]),
        ("voyage-4", [1, 400_000], [["0"], ["1"]]),
    ],
)
def test_embed_documents_batches_by_tokens(
    client_cls: Mock, model: str, counts: list[int], expected_batches: list[list[str]]
) -> None:
    client_cls.return_value.tokenize.return_value = _tokens(*counts)
    client_cls.return_value.embed.side_effect = _embed_by_text
    texts = [str(i) for i in range(len(counts))]
    result = VoyageAIEmbeddings(model=model).embed_documents(texts)
    sent = [c.args[0] for c in client_cls.return_value.embed.call_args_list]
    assert sent == expected_batches
    assert result == [[float(i)] for i in range(len(counts))]  # order kept


def test_no_token_counting_without_a_ceiling(client_cls: Mock) -> None:
    client_cls.return_value.embed.side_effect = _embed_by_text
    VoyageAIEmbeddings(model="voyage-code-4").embed_documents(["0", "1"])
    client_cls.return_value.tokenize.assert_not_called()


def test_rate_limit_retries_only_the_failed_batch(client_cls: Mock) -> None:
    client_cls.return_value.tokenize.return_value = _tokens(200_000, 200_000)
    embed = client_cls.return_value.embed
    embed.side_effect = [
        MagicMock(embeddings=[[0.0]]),
        RateLimitError("slow down"),
        MagicMock(embeddings=[[1.0]]),
    ]
    embedder = VoyageAIEmbeddings(
        model="voyage-4",
        rate_limit_handler=RetryRateLimitHandler(min_wait=0, jitter=False),
    )
    assert embedder.embed_documents(["0", "1"]) == [[0.0], [1.0]]
    assert [c.args[0] for c in embed.call_args_list] == [["0"], ["1"], ["1"]]


def test_embed_documents_batches_by_input_count(client_cls: Mock) -> None:
    client_cls.return_value.tokenize.return_value = _tokens(*[1] * 1001)
    client_cls.return_value.embed.side_effect = _embed_by_text
    texts = [str(i) for i in range(1001)]
    result = VoyageAIEmbeddings(model="voyage-4-lite").embed_documents(texts)
    sizes = [len(c.args[0]) for c in client_cls.return_value.embed.call_args_list]
    assert sizes == [1000, 1]
    assert len(result) == 1001


def test_token_count_failure_is_an_error(client_cls: Mock) -> None:
    client_cls.return_value.tokenize.side_effect = OSError("no tokenizer")
    with pytest.raises(EmbeddingsGenerationError, match="count tokens"):
        VoyageAIEmbeddings(model="voyage-4").embed_documents(["a"])
    client_cls.return_value.embed.assert_not_called()


def test_missing_embeddings_in_response_is_an_error(client_cls: Mock) -> None:
    client_cls.return_value.tokenize.return_value = _tokens(1, 1)
    client_cls.return_value.embed.return_value = MagicMock(embeddings=[[1.0]])
    embedder = VoyageAIEmbeddings(model="voyage-4")
    with pytest.raises(EmbeddingsGenerationError, match="expected 2 embeddings"):
        embedder.embed_documents(["a", "b"])


def test_api_error_detail_is_surfaced(client_cls: Mock) -> None:
    client_cls.return_value.embed.side_effect = InvalidRequestError("bad input")
    embedder = VoyageAIEmbeddings(model="voyage-4")
    with pytest.raises(
        EmbeddingsGenerationError, match="Failed to generate embedding with VoyageAI"
    ) as exc:
        embedder.embed_query("x")
    assert "bad input" in str(exc.value)
    # Not retried: only rate limits are.
    assert client_cls.return_value.embed.call_count == 1


def test_rate_limit_is_retried(client_cls: Mock) -> None:
    embed = client_cls.return_value.embed
    embed.side_effect = [RateLimitError("slow down")] * 3
    with pytest.raises(RetryError):
        VoyageAIEmbeddings(model="voyage-4").embed_query("x")
    assert embed.call_count == 3

    embed.reset_mock()
    embed.side_effect = [RateLimitError("slow down"), MagicMock(embeddings=[[1.0]])]
    assert VoyageAIEmbeddings(model="voyage-4").embed_query("x") == [1.0]


def _contextualized_response(*docs: list[str]) -> Any:
    return MagicMock(
        results=[
            MagicMock(
                index=i, embeddings=[[float(i)]] * len(chunks), chunk_texts=chunks
            )
            for i, chunks in enumerate(docs)
        ]
    )


def test_contextualized_documents_always_auto_chunk(client_cls: Mock) -> None:
    client_cls.return_value.tokenize.return_value = _tokens(5, 5)
    client_cls.return_value.contextualized_embed.return_value = (
        _contextualized_response(["a1", "a2"], ["b1"])
    )
    embedder = VoyageAIContextualizedEmbeddings(
        model="voyage-context-4", output_dimension=512
    )
    docs = embedder.embed_documents(["doc a", "doc b"])
    client_cls.return_value.contextualized_embed.assert_called_once_with(
        inputs=["doc a", "doc b"],
        model="voyage-context-4",
        output_dimension=512,
        input_type="document",
        enable_auto_chunking=True,
        chunk_size=32_000,
    )
    assert [d.chunks for d in docs] == [["a1", "a2"], ["b1"]]
    assert [len(d.embeddings) for d in docs] == [2, 1]


def test_contextualized_rejects_pre_chunked_documents(client_cls: Mock) -> None:
    embedder = VoyageAIContextualizedEmbeddings(model="voyage-context-3")
    with pytest.raises(ValueError, match="one string per document"):
        embedder.embed_documents([["chunk 1", "chunk 2"]])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="must not be empty"):
        embedder.embed_documents([])
    client_cls.return_value.contextualized_embed.assert_not_called()


def test_contextualized_response_without_chunk_text_is_an_error(
    client_cls: Mock,
) -> None:
    client_cls.return_value.tokenize.return_value = _tokens(1)
    client_cls.return_value.contextualized_embed.return_value = MagicMock(
        results=[MagicMock(index=0, embeddings=[[1.0]], chunk_texts=None)]
    )
    embedder = VoyageAIContextualizedEmbeddings(model="voyage-context-4")
    with pytest.raises(EmbeddingsGenerationError, match="text of every chunk"):
        embedder.embed_documents(["doc"])


def test_contextualized_query_never_auto_chunks(client_cls: Mock) -> None:
    client_cls.return_value.contextualized_embed.return_value = MagicMock(
        results=[MagicMock(index=0, embeddings=[[0.5, 0.25]])]
    )
    embedder = VoyageAIContextualizedEmbeddings(model="voyage-context-4")
    assert embedder.embed_query("question") == [0.5, 0.25]
    client_cls.return_value.contextualized_embed.assert_called_once_with(
        inputs=[["question"]], model="voyage-context-4", input_type="query"
    )


def test_contextualized_query_rejects_other_input_type(client_cls: Mock) -> None:
    embedder = VoyageAIContextualizedEmbeddings(model="voyage-context-4")
    with pytest.raises(ValueError, match="only supports input_type='query'"):
        embedder.embed_query("question", input_type="document")
    client_cls.return_value.contextualized_embed.assert_not_called()


def test_contextualized_documents_batch_by_tokens(client_cls: Mock) -> None:
    client_cls.return_value.tokenize.return_value = _tokens(70_000, 70_000, 40_000)
    client_cls.return_value.contextualized_embed.side_effect = lambda inputs, **_: (
        _contextualized_response(*[[d] for d in inputs])
    )
    embedder = VoyageAIContextualizedEmbeddings(model="voyage-context-3")
    docs = embedder.embed_documents(["d0", "d1", "d2"])
    sent = [
        c.kwargs["inputs"]
        for c in client_cls.return_value.contextualized_embed.call_args_list
    ]
    assert sent == [["d0"], ["d1", "d2"]]
    assert [d.chunks for d in docs] == [["d0"], ["d1"], ["d2"]]
