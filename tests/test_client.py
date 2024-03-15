import pytest
from neo4j_genai import GenAIClient
from neo4j_genai.types import Neo4jRecord
from unittest.mock import patch, MagicMock

def test_genai_client_supported_aura_version(driver):
    driver.execute_query.return_value = [[{"versions": ["5.11-aura"]}], None, None]

    GenAIClient(driver=driver)


def test_genai_client_no_supported_aura_version(driver):
    driver.execute_query.return_value = [[{"versions": ["5.3-aura"]}], None, None]

    with pytest.raises(ValueError) as excinfo:
        GenAIClient(driver=driver)

    assert "Version index is only supported in Neo4j version 5.11 or greater" in str(
        excinfo
    )


def test_genai_client_supported_version(driver):
    driver.execute_query.return_value = [[{"versions": ["5.11.5"]}], None, None]

    GenAIClient(driver=driver)


def test_genai_client_no_supported_version(driver):
    driver.execute_query.return_value = [[{"versions": ["4.3.5"]}], None, None]

    with pytest.raises(ValueError) as excinfo:
        GenAIClient(driver=driver)

    assert "Version index is only supported in Neo4j version 5.11 or greater" in str(
        excinfo
    )


def test_create_index_happy_path(driver, client):
    driver.execute_query.return_value = [None, None, None]
    create_query = (
        "CALL db.index.vector.createNodeIndex("
        "$name,"
        "$label,"
        "$property,"
        "toInteger($dimensions),"
        "$similarity_fn )"
    )

    client.create_index("my-index", "People", "name", 2048, "cosine")

    driver.execute_query.assert_called_once_with(
        create_query,
        {
            "name": "my-index",
            "label": "People",
            "property": "name",
            "dimensions": 2048,
            "similarity_fn": "cosine",
        },
    )


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

    records = client.similarity_search(name=index_name, query_vector=query_vector, top_k=top_k)

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

    records = client.similarity_search(name=index_name, query_text=query_text, top_k=top_k)

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
        client.similarity_search(name=index_name, query_text=query_text, top_k=top_k)


def test_similarity_search_both_text_and_vector(client):
    index_name = "my-index"
    query_text = "may thy knife chip and shatter"
    query_vector = [1.1, 2.2, 3.3]
    top_k = 5

    with pytest.raises(
        ValueError, match="You must provide exactly one of query_vector or query_text."
    ):
        client.similarity_search(
            name=index_name,
            query_text=query_text,
            query_vector=query_vector,
            top_k=top_k,
        )
