import pytest
from neo4j_genai import GenAIClient
from unittest.mock import Mock, patch
from neo4j.exceptions import CypherSyntaxError


@patch(
    "neo4j_genai.GenAIClient.database_query",
    return_value=[{"versions": ["5.11-aura"]}],
)
def test_genai_client_supported_aura_version(mock_database_query, driver):
    GenAIClient(driver)
    mock_database_query.assert_called_once()


@patch(
    "neo4j_genai.GenAIClient.database_query",
    return_value=[{"versions": ["5.3-aura"]}],
)
def test_genai_client_no_supported_aura_version(driver):
    with pytest.raises(ValueError):
        GenAIClient(driver)


@patch(
    "neo4j_genai.GenAIClient.database_query",
    return_value=[{"versions": ["5.11.5"]}],
)
def test_genai_client_supported_version(mock_database_query, driver):
    GenAIClient(driver)
    mock_database_query.assert_called_once()


@patch(
    "neo4j_genai.GenAIClient.database_query",
    return_value=[{"versions": ["4.3.5"]}],
)
def test_genai_client_no_supported_version(driver):
    with pytest.raises(ValueError):
        GenAIClient(driver)


@patch("neo4j_genai.GenAIClient.database_query")
def test_create_index_happy_path(mock_database_query, client):
    client.create_index("my-index", "People", "name", 2048, "cosine")
    query = (
        "CALL db.index.vector.createNodeIndex("
        "$name,"
        "$label,"
        "$property,"
        "toInteger($dimensions),"
        "$similarity_fn )"
    )
    mock_database_query.assert_called_once_with(
        query,
        params={
            "name": "my-index",
            "label": "People",
            "property": "name",
            "dimensions": 2048,
            "similarity_fn": "cosine",
        },
    )


def test_create_index_too_big_dimension(client):
    with pytest.raises(ValueError):
        client.create_index("my-index", "People", "name", 5024, "cosine")


def test_create_index_validation_error_dimensions(client):
    with pytest.raises(ValueError) as excinfo:
        client.create_index("my-index", "People", "name", "no-dim", "cosine")
    assert "Error for inputs to create_index" in str(excinfo)


def test_create_index_validation_error_similarity_fn(client):
    with pytest.raises(ValueError) as excinfo:
        client.create_index("my-index", "People", "name", 1536, "algebra")
    assert "Error for inputs to create_index" in str(excinfo)


@patch("neo4j_genai.GenAIClient.database_query")
def test_drop_index(mock_database_query, client):
    client.drop_index("my-index")

    query = "DROP INDEX $name"

    mock_database_query.assert_called_with(query, params={"name": "my-index"})


def test_database_query_happy(client, driver):
    class Session:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            pass

        def run(self, query, params):
            m_list = []
            for i in range(3):
                mock = Mock()
                mock.data.return_value = i
                m_list.append(mock)

            return m_list

    driver.session = Session
    res = client.database_query("MATCH (p:$label) RETURN p", {"label": "People"})
    assert res == [0, 1, 2]


def test_database_query_cypher_error(client, driver):
    class Session:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            pass

        def run(self, query, params):
            raise CypherSyntaxError

    driver.session = Session

    with pytest.raises(ValueError):
        client.database_query("MATCH (p:$label) RETURN p", {"label": "People"})


@patch("neo4j_genai.GenAIClient.database_query")
def test_similarity_search_vector_happy_path(mock_database_query, client):
    index_name = "my-index"
    query_vector = [1.1, 2.2, 3.3]
    top_k = 5

    client.similarity_search(name=index_name, query_vector=query_vector, top_k=top_k)

    query = """
        CALL db.index.vector.queryNodes($index_name, $top_k, $query_vector)
        YIELD node, score
        """
    mock_database_query.assert_called_once_with(
        query,
        params={
            "index_name": index_name,
            "top_k": top_k,
            "query_vector": query_vector,
        },
    )


@patch("neo4j_genai.GenAIClient.database_query")
def test_similarity_search_text_happy_path(mock_database_query, client_with_embedder):
    index_name = "my-index"
    query_text = "may thy knife chip and shatter"
    query_vector = [1.0 for _ in range(1536)]
    top_k = 5

    client_with_embedder.similarity_search(
        name=index_name, query_text=query_text, top_k=top_k
    )

    query = """
        CALL db.index.vector.queryNodes($index_name, $top_k, $query_vector)
        YIELD node, score
        """
    mock_database_query.assert_called_once_with(
        query,
        params={
            "index_name": index_name,
            "top_k": top_k,
            "query_vector": query_vector,
        },
    )


def test_similarity_search_missing_embedder_for_text(client):
    index_name = "my-index"
    query_text = "may thy knife chip and shatter"
    top_k = 5

    with pytest.raises(ValueError):
        client.similarity_search(name=index_name, query_text=query_text, top_k=top_k)


def test_similarity_search_both_text_and_vector(client):
    index_name = "my-index"
    query_text = "may thy knife chip and shatter"
    query_vector = [1.1, 2.2, 3.3]
    top_k = 5

    with pytest.raises(ValueError):
        client.similarity_search(
            name=index_name,
            query_text=query_text,
            query_vector=query_vector,
            top_k=top_k,
        )
