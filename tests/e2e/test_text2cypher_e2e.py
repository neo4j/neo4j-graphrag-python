import neo4j
import pytest
from neo4j_genai import Text2CypherRetriever
from unittest.mock import MagicMock


@pytest.mark.usefixtures("setup_neo4j_for_schema_query")
def test_t2c_retriever_search(driver: MagicMock, llm: MagicMock) -> None:
    t2c_query = """
    MATCH (a:LabelA {property_a: 'a'})-[:REL_TYPE]->(b:LabelB)
    RETURN a.property_a
    """
    retriever = Text2CypherRetriever(driver=driver, llm=llm)
    retriever.llm.invoke.return_value = t2c_query
    query_text = "dummy-text"
    results = retriever.search(query_text=query_text)
    assert isinstance(results, list)
    assert len(results) == 1
    for result in results:
        assert isinstance(result, neo4j.Record)
        assert "a.property_a" in result.keys()
