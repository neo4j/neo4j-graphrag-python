from neo4j_genai.generation.prompts import RagTemplate
from neo4j_genai.generation.rag import RAG
from neo4j_genai.types import RetrieverResult, RetrieverResultItem


def test_rag_prompt_template():
    template = RagTemplate()
    prompt = template.format(
        context="my context",
        query="user's query",
    )
    assert prompt == """Answer the user question using the following context

    Context:
    my context

    Question:
    user's query

    Answer:
    """


def test_rag_happy_path(driver, retriever_mock, llm):
    rag = RAG(
        retriever=retriever_mock,
        llm=llm,
    )
    retriever_mock.search.return_value = RetrieverResult(
        items=[RetrieverResultItem(content="item content 1")]
    )
    llm.invoke.return_value = "llm generated text"

    res = rag.search("question")

    retriever_mock.search.assert_called_once_with(
        query_text="question"
    )
    llm.invoke.assert_called_once_with( """Answer the user question using the following context

    Context:
    item content 1

    Question:
    question

    Answer:
    """)

    assert res == "llm generated text"
