from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import torch
from neo4j_graphrag.embeddings.base import Embedder
from neo4j_graphrag.embeddings.sentence_transformers import (
    SentenceTransformerEmbeddings,
)
from neo4j_graphrag.exceptions import EmbeddingsGenerationError


def get_mock_sentence_transformers() -> MagicMock:
    mock = MagicMock()
    # I know, I know... ¯\_(ツ)_/¯
    # This is to cover the if type checks in the embed_query method
    mock.Tensor = torch.Tensor
    mock.ndarray = np.ndarray
    return mock


@patch("builtins.__import__")
def test_initialization(mock_import: Mock) -> None:
    MockSentenceTransformer = get_mock_sentence_transformers()
    mock_import.return_value = MockSentenceTransformer
    instance = SentenceTransformerEmbeddings()
    MockSentenceTransformer.SentenceTransformer.assert_called_with("all-MiniLM-L6-v2")
    assert isinstance(instance, Embedder)


@patch("builtins.__import__")
def test_initialization_with_custom_model(mock_import: Mock) -> None:
    MockSentenceTransformer = get_mock_sentence_transformers()
    mock_import.return_value = MockSentenceTransformer
    custom_model = "distilbert-base-nli-stsb-mean-tokens"
    SentenceTransformerEmbeddings(model=custom_model)
    MockSentenceTransformer.SentenceTransformer.assert_called_with(custom_model)


@patch("builtins.__import__")
def test_embed_query(mock_import: Mock) -> None:
    MockSentenceTransformer = get_mock_sentence_transformers()
    mock_import.return_value = MockSentenceTransformer
    mock_model = MockSentenceTransformer.SentenceTransformer.return_value
    mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])

    instance = SentenceTransformerEmbeddings()
    result = instance.embed_query("test query")

    mock_model.encode.assert_called_with(["test query"])
    assert isinstance(result, list)
    assert result == [0.1, 0.2, 0.3]


@patch("builtins.__import__", side_effect=ImportError)
def test_import_error(mock_import: Mock) -> None:
    with pytest.raises(ImportError):
        SentenceTransformerEmbeddings()


@patch("builtins.__import__")
def test_embed_query_error_handling(mock_import: Mock) -> None:
    MockSentenceTransformer = get_mock_sentence_transformers()
    mock_import.return_value = MockSentenceTransformer
    mock_model = MockSentenceTransformer.SentenceTransformer.return_value
    mock_model.encode.side_effect = Exception("Model error")

    instance = SentenceTransformerEmbeddings()
    with pytest.raises(
        EmbeddingsGenerationError,
        match="Failed to generate embedding with SentenceTransformer",
    ):
        instance.embed_query("test query")
