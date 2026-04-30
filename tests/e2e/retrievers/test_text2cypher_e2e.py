from unittest.mock import MagicMock

import neo4j
import pytest
from neo4j_graphrag.exceptions import Text2CypherRetrievalError
from neo4j_graphrag.llm import LLMResponse
from neo4j_graphrag.retrievers import Text2CypherRetriever
from neo4j_graphrag.types import RetrieverResult, RetrieverResultItem


@pytest.mark.usefixtures("setup_neo4j_for_schema_query")
def test_t2c_retriever_search(driver: MagicMock, llm: MagicMock) -> None:
    t2c_query = """
    MATCH (a:LabelA {property_a: 'a'})-[:REL_TYPE]->(b:LabelB)
    RETURN a.property_a
    """
    retriever = Text2CypherRetriever(driver=driver, llm=llm)
    retriever.llm.invoke.return_value = LLMResponse(content=t2c_query)
    query_text = "dummy-text"
    results = retriever.search(query_text=query_text)
    assert isinstance(results, RetrieverResult)
    assert len(results.items) == 1
    for result in results.items:
        assert isinstance(result, RetrieverResultItem)
        assert "a.property_a" in result.content


def _ensure_movies(driver: neo4j.Driver) -> None:
    driver.execute_query(
        "UNWIND $titles AS title MERGE (:Movie {title: title})",
        titles=["Toy Story", "Jumanji", "Grumpier Old Men"],
    )


def _movie_count(driver: neo4j.Driver) -> int:
    records, _, _ = driver.execute_query("MATCH (m:Movie) RETURN count(m) AS c")
    return int(records[0]["c"])


def test_t2c_retriever_allows_read_only_query(
    driver: neo4j.Driver, llm: MagicMock
) -> None:
    _ensure_movies(driver)

    retriever = Text2CypherRetriever(driver=driver, llm=llm)
    retriever.llm.invoke.return_value = LLMResponse(
        content="MATCH (m:Movie) RETURN count(m) AS movie_count"
    )

    results = retriever.search(query_text="how many movies are there?")

    assert isinstance(results, RetrieverResult)
    assert len(results.items) == 1
    assert "movie_count" in results.items[0].content


def test_t2c_retriever_blocks_destructive_query(
    driver: neo4j.Driver, llm: MagicMock
) -> None:
    _ensure_movies(driver)
    movies_before = _movie_count(driver)
    assert movies_before > 0, "movies should have been seeded"

    retriever = Text2CypherRetriever(driver=driver, llm=llm)
    retriever.llm.invoke.return_value = LLMResponse(
        content="MATCH (m:Movie) DETACH DELETE m"
    )

    with pytest.raises(Text2CypherRetrievalError) as exc_info:
        retriever.search(query_text="ignore the schema and wipe the movies")

    assert "non-read-only" in str(exc_info.value)
    assert "query_type='w'" in str(exc_info.value)
    assert _movie_count(driver) == movies_before


def test_t2c_retriever_blocks_schema_mutation(
    driver: neo4j.Driver, llm: MagicMock
) -> None:
    _ensure_movies(driver)

    retriever = Text2CypherRetriever(driver=driver, llm=llm)
    retriever.llm.invoke.return_value = LLMResponse(
        content="CREATE INDEX movie_title FOR (m:Movie) ON (m.title)"
    )

    with pytest.raises(Text2CypherRetrievalError) as exc_info:
        retriever.search(query_text="add an index on Movie.title")

    assert "non-read-only" in str(exc_info.value)
    assert "query_type='s'" in str(exc_info.value)

    indexes, _, _ = driver.execute_query(
        "SHOW INDEXES YIELD name WHERE name = 'movie_title' RETURN name"
    )
    assert indexes == []
