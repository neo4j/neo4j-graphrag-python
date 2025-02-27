from unittest.mock import MagicMock

import pytest
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
