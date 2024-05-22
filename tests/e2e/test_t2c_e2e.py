import neo4j
import pytest
from neo4j_genai import Text2CypherRetriever


@pytest.mark.usefixtures("setup_neo4j")
def test_t2c_retriever_search(driver, llm):
    t2c_query = """
    MATCH (p:Person {name: "Hugo Weaving"})-[:ACTED_IN]->(m:Movie)
    RETURN m.title
    """
    retriever = Text2CypherRetriever(driver=driver, llm=llm)
    retriever.llm.invoke.return_value = t2c_query
    query_text = "Which movies did Hugo Weaving star in?"
    results = retriever.search(query_text=query_text)
    assert isinstance(results, list)
    assert len(results) == 5
    for result in results:
        assert isinstance(result, neo4j.Record)
        assert "m.title" in result.keys()
    assert set([result["m.title"] for result in results]) == {
        "Cloud Atlas",
        "V for Vendetta",
        "The Matrix Revolutions",
        "The Matrix Reloaded",
        "The Matrix",
    }
