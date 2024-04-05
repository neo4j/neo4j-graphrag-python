import pytest
from unittest.mock import patch, MagicMock
from neo4j_genai import VectorRetriever
from neo4j_genai.types import Neo4jRecord


def test_vector_retriever_supported_aura_version(driver):
    driver.execute_query.return_value = [[{"versions": ["5.18-aura"]}], None, None]

    VectorRetriever(driver=driver, index_name="my-index")


def test_vector_retriever_no_supported_aura_version(driver):
    driver.execute_query.return_value = [[{"versions": ["5.3-aura"]}], None, None]

    with pytest.raises(ValueError) as excinfo:
        VectorRetriever(driver=driver, index_name="my-index")

    assert "This package only supports Neo4j version 5.18.1 or greater" in str(excinfo)


def test_vector_retriever_supported_version(driver):
    driver.execute_query.return_value = [[{"versions": ["5.19.0"]}], None, None]

    VectorRetriever(driver=driver, index_name="my-index")


def test_vector_retriever_no_supported_version(driver):
    driver.execute_query.return_value = [[{"versions": ["4.3.5"]}], None, None]

    with pytest.raises(ValueError) as excinfo:
        VectorRetriever(driver=driver, index_name="my-index")

    assert "This package only supports Neo4j version 5.18.1 or greater" in str(excinfo)


@patch("neo4j_genai.VectorRetriever._verify_version")
def test_similarity_search_vector_happy_path(_verify_version_mock, driver):
    custom_embeddings = MagicMock()

    index_name = "my-index"
    dimensions = 1536
    query_vector = [1.0 for _ in range(dimensions)]
    top_k = 5

    retriever = VectorRetriever(driver, index_name, custom_embeddings)

    retriever.driver.execute_query.return_value = [
        [{"node": "dummy-node", "score": 1.0}],
        None,
        None,
    ]
    search_query = """
        CALL db.index.vector.queryNodes($index_name, $top_k, $query_vector)
        YIELD node, score
        """

    records = retriever.search(query_vector=query_vector, top_k=top_k)

    custom_embeddings.embed_query.assert_not_called()

    retriever.driver.execute_query.assert_called_once_with(
        search_query,
        {
            "index_name": index_name,
            "top_k": top_k,
            "query_vector": query_vector,
        },
    )

    assert records == [Neo4jRecord(node="dummy-node", score=1.0)]


@patch("neo4j_genai.VectorRetriever._verify_version")
def test_similarity_search_text_happy_path(_verify_version_mock, driver):
    embed_query_vector = [1.0 for _ in range(1536)]
    custom_embeddings = MagicMock()
    custom_embeddings.embed_query.return_value = embed_query_vector

    index_name = "my-index"
    query_text = "may thy knife chip and shatter"
    top_k = 5

    retriever = VectorRetriever(driver, index_name, custom_embeddings)

    driver.execute_query.return_value = [
        [{"node": "dummy-node", "score": 1.0}],
        None,
        None,
    ]

    search_query = """
        CALL db.index.vector.queryNodes($index_name, $top_k, $query_vector)
        YIELD node, score
        """

    records = retriever.search(query_text=query_text, top_k=top_k)

    custom_embeddings.embed_query.assert_called_once_with(query_text)

    driver.execute_query.assert_called_once_with(
        search_query,
        {
            "index_name": index_name,
            "top_k": top_k,
            "query_vector": embed_query_vector,
        },
    )

    assert records == [Neo4jRecord(node="dummy-node", score=1.0)]


def test_similarity_search_missing_embedder_for_text(retriever):
    query_text = "may thy knife chip and shatter"
    top_k = 5

    with pytest.raises(ValueError, match="Embedding method required for text query"):
        retriever.search(query_text=query_text, top_k=top_k)


def test_similarity_search_both_text_and_vector(retriever):
    query_text = "may thy knife chip and shatter"
    query_vector = [1.1, 2.2, 3.3]
    top_k = 5

    with pytest.raises(
        ValueError, match="You must provide exactly one of query_vector or query_text."
    ):
        retriever.search(
            query_text=query_text,
            query_vector=query_vector,
            top_k=top_k,
        )


@patch("neo4j_genai.VectorRetriever._verify_version")
def test_similarity_search_vector_bad_results(_verify_version_mock, driver):
    custom_embeddings = MagicMock()

    index_name = "my-index"
    dimensions = 1536
    query_vector = [1.0 for _ in range(dimensions)]
    top_k = 5

    retriever = VectorRetriever(driver, index_name, custom_embeddings)

    retriever.driver.execute_query.return_value = [
        [{"node": "dummy-node", "score": "adsa"}],
        None,
        None,
    ]
    search_query = """
        CALL db.index.vector.queryNodes($index_name, $top_k, $query_vector)
        YIELD node, score
        """

    with pytest.raises(ValueError):
        retriever.search(query_vector=query_vector, top_k=top_k)

    custom_embeddings.embed_query.assert_not_called()

    retriever.driver.execute_query.assert_called_once_with(
        search_query,
        {
            "index_name": index_name,
            "top_k": top_k,
            "query_vector": query_vector,
        },
    )
