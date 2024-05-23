import pytest
from unittest.mock import patch
from neo4j_genai.embeddings import SentenceTransformerEmbeddings
from neo4j_genai.embedder import Embedder


@patch("sentence_transformers.SentenceTransformer")
def test_initialization(MockSentenceTransformer):
    instance = SentenceTransformerEmbeddings()
    MockSentenceTransformer.assert_called_with("all-MiniLM-L6-v2")
    assert isinstance(instance, Embedder)


@patch("sentence_transformers.SentenceTransformer")
def test_initialization_with_custom_model(MockSentenceTransformer):
    custom_model = "distilbert-base-nli-stsb-mean-tokens"
    SentenceTransformerEmbeddings(model=custom_model)
    MockSentenceTransformer.assert_called_with(custom_model)


@patch("sentence_transformers.SentenceTransformer")
def test_embed_query(MockSentenceTransformer):
    mock_model = MockSentenceTransformer.return_value
    mock_model.encode.return_value = [0.1, 0.2, 0.3]

    instance = SentenceTransformerEmbeddings()
    result = instance.embed_query("test query")

    mock_model.encode.assert_called_with(["test query"])
    assert result == [0.1, 0.2, 0.3]


@patch("sentence_transformers.SentenceTransformer", side_effect=ImportError)
def test_import_error(MockSentenceTransformer):
    with pytest.raises(ImportError):
        SentenceTransformerEmbeddings()
