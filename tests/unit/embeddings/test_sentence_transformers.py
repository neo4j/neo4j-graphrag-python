from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from neo4j_graphrag.embeddings.base import Embedder
from neo4j_graphrag.embeddings.sentence_transformers import (
    SentenceTransformerEmbeddings,
)


@patch("neo4j_graphrag.embeddings.sentence_transformers.sentence_transformers")
def test_initialization(MockSentenceTransformer: MagicMock) -> None:
    instance = SentenceTransformerEmbeddings()
    MockSentenceTransformer.SentenceTransformer.assert_called_with("all-MiniLM-L6-v2")
    assert isinstance(instance, Embedder)


@patch("neo4j_graphrag.embeddings.sentence_transformers.sentence_transformers")
def test_initialization_with_custom_model(MockSentenceTransformer: MagicMock) -> None:
    custom_model = "distilbert-base-nli-stsb-mean-tokens"
    SentenceTransformerEmbeddings(model=custom_model)
    MockSentenceTransformer.SentenceTransformer.assert_called_with(custom_model)


@patch("neo4j_graphrag.embeddings.sentence_transformers.sentence_transformers")
def test_embed_query(MockSentenceTransformer: MagicMock) -> None:
    mock_model = MockSentenceTransformer.SentenceTransformer.return_value
    mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])

    instance = SentenceTransformerEmbeddings()
    result = instance.embed_query("test query")

    mock_model.encode.assert_called_with(
        ["test query"]
    )
    assert isinstance(result, list)
    assert result == [0.1, 0.2, 0.3]


@patch(
    "neo4j_graphrag.embeddings.sentence_transformers.sentence_transformers",
    None,
)
def test_import_error() -> None:
    with pytest.raises(ImportError):
        SentenceTransformerEmbeddings()
