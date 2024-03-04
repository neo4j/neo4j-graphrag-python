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
        client.create_index("my-index", "People", "name", "no-dim", "algebra")
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


def test_similarity_search():
    pass
