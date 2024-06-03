import pytest

from neo4j_genai import VectorRetriever
from neo4j_genai.generation.rag import RAG


@pytest.mark.usefixtures("setup_neo4j_for_retrieval")
def test_rag_happy_path(driver, custom_embedder, llm):
    retriever = VectorRetriever(
        driver, "vector-index-name", custom_embedder
    )
    rag = RAG(
        retriever=retriever,
        llm=llm,
    )
    rag.llm.invoke.return_value = "some text"

    result = rag.search(
        query="Find me a book about Fremen",
        retriever_config={
            "top_k": 5,
        }
    )

    llm.invoke.assert_called_once()
    assert isinstance(result, str)
    assert result == "some text"
