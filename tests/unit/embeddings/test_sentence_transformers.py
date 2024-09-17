from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from neo4j_graphrag.embeddings.embedder import Embedder
from neo4j_graphrag.embeddings.sentence_transformers import (
    SentenceTransformerEmbeddings,
)


@patch("sentence_transformers.SentenceTransformer")
def test_initialization(MockSentenceTransformer: MagicMock) -> None:
    instance = SentenceTransformerEmbeddings()
    MockSentenceTransformer.assert_called_with("all-MiniLM-L6-v2")
    assert isinstance(instance, Embedder)


@patch("sentence_transformers.SentenceTransformer")
def test_initialization_with_custom_model(MockSentenceTransformer: MagicMock) -> None:
    custom_model = "distilbert-base-nli-stsb-mean-tokens"
    SentenceTransformerEmbeddings(model=custom_model)
    MockSentenceTransformer.assert_called_with(custom_model)


@patch("sentence_transformers.SentenceTransformer")
def test_embed_query(MockSentenceTransformer: MagicMock) -> None:
    mock_model = MockSentenceTransformer.return_value
    mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])

    instance = SentenceTransformerEmbeddings()
    result = instance.embed_query("test query")

    mock_model.encode.assert_called_with(["test query"])
    assert result == [0.1, 0.2, 0.3]
    assert isinstance(result, list)


@patch("sentence_transformers.SentenceTransformer", side_effect=ImportError)
def test_import_error(MockSentenceTransformer: MagicMock) -> None:
    with pytest.raises(ImportError):
        SentenceTransformerEmbeddings()
