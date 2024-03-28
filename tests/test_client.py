import pytest
from neo4j_genai import GenAIClient
from unittest.mock import patch, MagicMock
from neo4j.exceptions import CypherSyntaxError

from neo4j_genai.types import Neo4jRecord


def test_genai_client_supported_aura_version(driver):
    driver.execute_query.return_value = [[{"versions": ["5.18-aura"]}], None, None]

    GenAIClient(driver=driver)


def test_genai_client_no_supported_aura_version(driver):
    driver.execute_query.return_value = [[{"versions": ["5.3-aura"]}], None, None]

    with pytest.raises(ValueError) as excinfo:
        GenAIClient(driver=driver)

    assert "This package only supports Neo4j version 5.18.1 or greater" in str(excinfo)


def test_genai_client_supported_version(driver):
    driver.execute_query.return_value = [[{"versions": ["5.19.0"]}], None, None]

    GenAIClient(driver=driver)


def test_genai_client_no_supported_version(driver):
    driver.execute_query.return_value = [[{"versions": ["4.3.5"]}], None, None]

    with pytest.raises(ValueError) as excinfo:
        GenAIClient(driver=driver)

    assert "This package only supports Neo4j version 5.18.1 or greater" in str(excinfo)


def test_create_index_happy_path(driver, client):
    driver.execute_query.return_value = [None, None, None]
    create_query = (
        "CREATE VECTOR INDEX $name IF NOT EXISTS FOR (n:People) ON n.name OPTIONS "
        "{ indexConfig: { `vector.dimensions`: toInteger($dimensions), `vector.similarity_function`: $similarity_fn } }"
    )

    client.create_index("my-index", "People", "name", 2048, "cosine")

    driver.execute_query.assert_called_once_with(
        create_query,
        {"name": "my-index", "dimensions": 2048, "similarity_fn": "cosine"},
    )


def test_create_index_ensure_escaping(driver, client):
    driver.execute_query.return_value = [None, None, None]
    create_query = (
        "CREATE VECTOR INDEX $name IF NOT EXISTS FOR (n:People) ON n.name OPTIONS "
        "{ indexConfig: { `vector.dimensions`: toInteger($dimensions), `vector.similarity_function`: $similarity_fn } }"
    )

    client.create_index("my-complicated-`-index", "People", "name", 2048, "cosine")

    driver.execute_query.assert_called_once_with(
        create_query,
        {
            "name": "my-complicated-`-index",
            "dimensions": 2048,
            "similarity_fn": "cosine",
        },
    )


def test_create_index_validation_error_dimensions_negative_integer(client):
    with pytest.raises(ValueError) as excinfo:
        client.create_index("my-index", "People", "name", -5, "cosine")
    assert "Error for inputs to create_index" in str(excinfo)


def test_create_index_validation_error_dimensions(client):
    with pytest.raises(ValueError) as excinfo:
        client.create_index("my-index", "People", "name", "no-dim", "cosine")
    assert "Error for inputs to create_index" in str(excinfo)


def test_create_index_validation_error_similarity_fn(client):
    with pytest.raises(ValueError) as excinfo:
        client.create_index("my-index", "People", "name", 1536, "algebra")
    assert "Error for inputs to create_index" in str(excinfo)


def test_drop_index(client):
    client.driver.execute_query.return_value = [None, None, None]
    drop_query = "DROP INDEX $name"

    client.drop_index("my-index")

    client.driver.execute_query.assert_called_once_with(
        drop_query,
        {"name": "my-index"},
    )


@patch("neo4j_genai.GenAIClient._verify_version")
def test_similarity_search_vector_happy_path(_verify_version_mock, driver):
    custom_embeddings = MagicMock()

    client = GenAIClient(driver, custom_embeddings)

    index_name = "my-index"
    dimensions = 1536
    query_vector = [1.0 for _ in range(dimensions)]
    top_k = 5

    client.driver.execute_query.return_value = [
        [{"node": "dummy-node", "score": 1.0}],
        None,
        None,
    ]
    search_query = """
        CALL db.index.vector.queryNodes($index_name, $top_k, $query_vector)
        YIELD node, score
        """

    records = client.search_similar_vectors(
        name=index_name, query_vector=query_vector, top_k=top_k
    )

    custom_embeddings.embed_query.assert_not_called()

    client.driver.execute_query.assert_called_once_with(
        search_query,
        {
            "index_name": index_name,
            "top_k": top_k,
            "query_vector": query_vector,
        },
    )

    assert records == [Neo4jRecord(node="dummy-node", score=1.0)]


@patch("neo4j_genai.GenAIClient._verify_version")
def test_similarity_search_text_happy_path(_verify_version_mock, driver):
    embed_query_vector = [1.0 for _ in range(1536)]
    custom_embeddings = MagicMock()
    custom_embeddings.embed_query.return_value = embed_query_vector

    client = GenAIClient(driver, custom_embeddings)

    index_name = "my-index"
    query_text = "may thy knife chip and shatter"
    top_k = 5

    driver.execute_query.return_value = [
        [{"node": "dummy-node", "score": 1.0}],
        None,
        None,
    ]

    search_query = """
        CALL db.index.vector.queryNodes($index_name, $top_k, $query_vector)
        YIELD node, score
        """

    records = client.search_similar_vectors(
        name=index_name, query_text=query_text, top_k=top_k
    )

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


def test_similarity_search_missing_embedder_for_text(client):
    index_name = "my-index"
    query_text = "may thy knife chip and shatter"
    top_k = 5

    with pytest.raises(ValueError, match="Embedding method required for text query"):
        client.search_similar_vectors(
            name=index_name, query_text=query_text, top_k=top_k
        )


def test_similarity_search_both_text_and_vector(client):
    index_name = "my-index"
    query_text = "may thy knife chip and shatter"
    query_vector = [1.1, 2.2, 3.3]
    top_k = 5

    with pytest.raises(
        ValueError, match="You must provide exactly one of query_vector or query_text."
    ):
        client.search_similar_vectors(
            name=index_name,
            query_text=query_text,
            query_vector=query_vector,
            top_k=top_k,
        )


@patch("neo4j_genai.GenAIClient._verify_version")
def test_similarity_search_vector_bad_results(_verify_version_mock, driver):
    custom_embeddings = MagicMock()

    client = GenAIClient(driver, custom_embeddings)

    index_name = "my-index"
    dimensions = 1536
    query_vector = [1.0 for _ in range(dimensions)]
    top_k = 5

    client.driver.execute_query.side_effect = ValueError
    search_query = """
        CALL db.index.vector.queryNodes($index_name, $top_k, $query_vector)
        YIELD node, score
        """

    with pytest.raises(ValueError):
        client.search_similar_vectors(
            name=index_name, query_vector=query_vector, top_k=top_k
        )

    custom_embeddings.embed_query.assert_not_called()

    client.driver.execute_query.assert_called_once_with(
        search_query,
        {
            "index_name": index_name,
            "top_k": top_k,
            "query_vector": query_vector,
        },
    )


@patch("neo4j_genai.GenAIClient._verify_version")
def test_custom_retrieval_query_happy_path(_verify_version_mock, driver):
    embed_query_vector = [1.0 for _ in range(1536)]
    custom_embeddings = MagicMock()
    custom_embeddings.embed_query.return_value = embed_query_vector

    client = GenAIClient(driver, custom_embeddings)

    index_name = "my-index"
    query_text = "may thy knife chip and shatter"
    top_k = 5

    driver.execute_query.return_value = [
        [{"node_id": 123, "text": "dummy-text", "score": 1.0}],
        None,
        None,
    ]

    search_query = """
        CALL db.index.vector.queryNodes($index_name, $top_k, $query_vector)
        YIELD node, score
        """
    custom_retrieval_query = """
        RETURN node.id AS node_id, node.text AS text, score
        """

    records = client.custom_search_similar_vectors(
        name=index_name,
        query_text=query_text,
        top_k=top_k,
        custom_retrieval_query=custom_retrieval_query,
    )

    custom_embeddings.embed_query.assert_called_once_with(query_text)

    driver.execute_query.assert_called_once_with(
        search_query + custom_retrieval_query,
        {
            "index_name": index_name,
            "top_k": top_k,
            "query_vector": embed_query_vector,
        },
    )

    assert records == [{"node_id": 123, "text": "dummy-text", "score": 1.0}]


@patch("neo4j_genai.GenAIClient._verify_version")
def test_custom_retrieval_query_with_params(_verify_version_mock, driver):
    embed_query_vector = [1.0 for _ in range(1536)]
    custom_embeddings = MagicMock()
    custom_embeddings.embed_query.return_value = embed_query_vector

    client = GenAIClient(driver, custom_embeddings)

    index_name = "my-index"
    query_text = "may thy knife chip and shatter"
    top_k = 5

    driver.execute_query.return_value = [
        [{"node_id": 123, "text": "dummy-text", "score": 1.0}],
        None,
        None,
    ]

    search_query = """
        CALL db.index.vector.queryNodes($index_name, $top_k, $query_vector)
        YIELD node, score
        """
    custom_retrieval_query = """
        RETURN node.id AS node_id, node.text AS text, score, {test: $param} AS metadata
        """
    custom_params = {
        "param": "dummy-param",
    }

    records = client.custom_search_similar_vectors(
        name=index_name,
        query_text=query_text,
        top_k=top_k,
        custom_retrieval_query=custom_retrieval_query,
        custom_params=custom_params,
    )

    custom_embeddings.embed_query.assert_called_once_with(query_text)

    driver.execute_query.assert_called_once_with(
        search_query + custom_retrieval_query,
        {
            "index_name": index_name,
            "top_k": top_k,
            "query_vector": embed_query_vector,
            "param": "dummy-param",
        },
    )

    assert records == [{"node_id": 123, "text": "dummy-text", "score": 1.0}]


@patch("neo4j_genai.GenAIClient._verify_version")
def test_custom_retrieval_query_cypher_error(_verify_version_mock, driver):
    embed_query_vector = [1.0 for _ in range(1536)]
    custom_embeddings = MagicMock()
    custom_embeddings.embed_query.return_value = embed_query_vector

    client = GenAIClient(driver, custom_embeddings)

    index_name = "my-index"
    query_text = "may thy knife chip and shatter"
    top_k = 5

    driver.execute_query.side_effect = CypherSyntaxError

    custom_retrieval_query = """
        this is not a cypher query
        """

    with pytest.raises(CypherSyntaxError):
        client.custom_search_similar_vectors(
            name=index_name,
            query_text=query_text,
            top_k=top_k,
            custom_retrieval_query=custom_retrieval_query,
        )
