.. _rag-documentation:

RAG Documentation
#################

************************************
Retrieval-Augmented Generation (RAG)
************************************
RAG is a technique that enhances Large Language Model (LLM) responses by retrieving
source information from external data stores to augment generated responses.

This package enables Python developers to perform RAG using Neo4j.


************************************
Overview
************************************

.. code:: python

    from neo4j import GraphDatabase
    from neo4j_genai import VectorRetriever, OpenAILLM, RAG

    URI = "neo4j://localhost:7687"
    AUTH = ("neo4j", "password")

    INDEX_NAME = "embedding-name"

    # Connect to Neo4j database
    driver = GraphDatabase.driver(URI, auth=AUTH)

    # Create Embedder object
    embedder = OpenAIEmbeddings(model="text-embedding-3-large")

    # Initialize the retriever
    retriever = VectorRetriever(driver, INDEX_NAME, embedder)

    # Initialize the LLM
    # Note: the OPENAI_API_KEY must be in the env vars
    llm = OpenAILLM(model_name="gpt-4o", model_params={"temperature": 0})

    # Initialize the RAG pipeline
    rag = RAG(retriever=vector_retriever, llm=llm)

    # Query the graph
    query_text = "How do I do similarity search in Neo4j?"
    response = rag.search(query_text=query_text, retriever_config={"top_k": 5})


The retriever can be any of the :ref:`supported retrievers<retrievers>`, or any class
inheriting from the `Retriever` interface.

***************
Advanced usage
***************


Using another LLM
==================

This package only provide support for OpenAI LLM. If you need to use another LLM,
you need to subclass the `LLMInterface`:

.. autoclass:: neo4j_genai.llm.LLMInterface
    :members:
    :show-inheritance:

Configuring the prompt
=======================

Prompt are managed through `PromptTemplate` classes. More
specifically, the `RAG` pipeline uses a `RagTemplate` with
a default prompt. You can use another prompt by subclassing
the `RagTemplate` class and passing it to the `RAG` pipeline
object during initialization:

.. code:: python

    from neo4j_genai import RagTemplate, RAG

    # ...

    prompt_template = RagTemplate(
        prompt="Answer the question {question} using context {context} and examples {examples}",
        expected_inputs=["context", "question", "examples"]
    )
    rag = RAG(retriever=vector_retriever, llm=llm, prompt_template=prompt_template)

    # ...

For more details, see:

.. autoclass:: neo4j_genai.generation.prompts.PromptTemplate
    :members:

and

.. autoclass:: neo4j_genai.generation.prompts.RagTemplate
    :members:
