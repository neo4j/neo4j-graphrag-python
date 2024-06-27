.. _user-guide:

User Guide
#################

This guide help you getting started with the Neo4j GenAI Python package,
and explain how to configure it to meet your requirements.


************
Quick start
************

In order to perform a GraphRAG query using the neo4j-genai package, you need to
instantiate a few components:

1. A neo4j driver: used to query your Neo4j database
2. A Retriever: the neo4j-genai package provides some implementations (see ???) and lets you write your own if none of the provided implementations matches your needs (see ???)
3. An LLM: in order to generate the answer, we need to call an LLM model. The neo4j-genai package currently only provides implementation for the OpenAI LLMs, but our interface is compatible with LangChain.

In practice, it's done with only a few lines of code:

.. code:: python

    from neo4j import GraphDatabase
    from neo4j_genai.retrievers import VectorRetriever
    from neo4j_genai.llm import OpenAILLM
    from neo4j_genai.generation import GraphRAG
    from neo4j_genai.embeddings.openai import OpenAIEmbeddings

    # 1. Neo4j driver
    URI = "neo4j://localhost:7687"
    AUTH = ("neo4j", "password")

    INDEX_NAME = "index-name"

    # Connect to Neo4j database
    driver = GraphDatabase.driver(URI, auth=AUTH)

    # 2. Retriever
    # Create Embedder object, needed to convert the user question (text) to a vector
    embedder = OpenAIEmbeddings(model="text-embedding-3-large")

    # Initialize the retriever
    retriever = VectorRetriever(driver, INDEX_NAME, embedder)

    # 3. LLM
    # Note: the OPENAI_API_KEY must be in the env vars
    llm = OpenAILLM(model_name="gpt-4o", model_params={"temperature": 0})

    # Initialize the RAG pipeline
    rag = GraphRAG(retriever=retriever, llm=llm)

    # Query the graph
    query_text = "How do I do similarity search in Neo4j?"
    response = rag.search(query_text=query_text, retriever_config={"top_k": 5})
    print(response.answer)


Let's dig into more details and learn how we can customize this code.

******************************
GraphRAG Configuration
******************************

Using another LLM model
========================

If you do not wish to use OpenAI, you have two available options:

1. Use any LangChain chat model
2. Implement your own interface

Both options are illustrated below, using a local Ollama model as an example.

Using a model from LangChain
-----------------------------

The LangChain python package contains implementations for many different LLMs
and providers. Their interface is compatible with our GraphRAG interface, so you
can use them easily:

.. code:: python

    from neo4j_genai.generation import GraphRAG
    from langchain_community.chat_models import ChatOllama

    # retriever = ...

    llm = ChatOllama(model="llama3:8b")
    rag = GraphRAG(retriever=retriever, llm=llm)
    query_text = "How do I do similarity search in Neo4j?"
    response = rag.search(query_text=query_text, retriever_config={"top_k": 5})
    print(response.answer)


Using a custom model
-----------------------------

If you do not wish to use LangChain, you can create your own LLM class by subclassing
the LLMInterface. Here is an example using the Python ollama client:

.. code:: python

    import ollama
    from neo4j_genai.llm import LLMInterface, LLMResponse

    class OllamaLLM(LLMInterface):

        def invoke(self, input: str) -> LLMResponse:
            response = ollama.chat(model=self.model_name, messages=[
              {
                'role': 'user',
                'content': input,
              },
            ])
            return LLMResponse(
                content=response["message"]["content"]
            )

    # retriever = ...

    llm = OllamaLLM("llama3:8b")

    rag = GraphRAG(retriever=retriever, llm=llm)
    query_text = "How do I do similarity search in Neo4j?"
    response = rag.search(query_text=query_text, retriever_config={"top_k": 5})
    print(response.answer)

Also see :ref:`llminterface`.


Configuring the prompt
========================

Prompt are managed through `PromptTemplate` classes. More
specifically, the `GraphRAG` pipeline uses a `RagTemplate` with
a default prompt. You can use another prompt by subclassing
the `RagTemplate` class and passing it to the `GraphRAG` pipeline
object during initialization:

.. code:: python

    from neo4j_genai.generation import RagTemplate, GraphRAG

    # retriever = ...
    # llm = ...

    prompt_template = RagTemplate(
        prompt="Answer the question {question} using context {context} and examples {examples}",
        expected_inputs=["context", "question", "examples"]
    )

    rag = GraphRAG(retriever=retriever, llm=llm, prompt_template=prompt_template)

    # ...


Also see :ref:`prompttemplate`.


The last piece you can configure in the GraphRAG pipeline is the retriever. The different options
are described below.


************************
Retriever Configuration
************************

We provide implementation for the most commonly used retrievers:

1. Vector retriever: performs a similarity search based on a Neo4j vector index and a query text or vector. Returns the matched node.
2. Vector Cypher retriever: performs a similarity search based on a Neo4j vector index and a query text or vector. Returned results can be configured through a retrieval query parameter that will be executed after the index search. It can be used to fetch more context around the matched node.
3. External retrievers: use these retrievers when your vectors are not saved in Neo4j but in an external vector database. We currently support Weaviate and Pinecone vector databases.
4. Hybrid and Hybrid cypher retrievers: these retrievers use both a vector and full-text indexes.

Retrievers all expose a `search` method that we will discuss in the next sections.

Also see :ref:`retrievers`.

Vector retriever
===================

The easiest way to instantiate a vector retriever is:

.. code:: python

    from neo4j_genai.retrievers import VectorRetriever

    retriever = VectorRetriever(
        driver,
        index_name=POSTER_INDEX_NAME,
    )

The `index_name` is the name of the Neo4j vector index that will be used for similarity search.


Search similar vector
-----------------------------

To find the top 3 most similar nodes, you can perform a search by vector:

.. code:: python

    vector = []  # a list of floats, same size as the vectors in the Neo4j vector index
    retriever_result = retriever.search(query_vector=vector, top_k=3)

However, most of the time, you will be given a text (from user) and not a vector. This
use case is covered thanks to the `Embedder`.

Search similar text
-----------------------------

When searching for a text, you must tell the retriever how to transform (embbed) the text
to a vector. This is the reason why an embedder is required when you initialize the retriever:

.. code:: python

    embedder = OpenAIEmbeddings(model="text-embedding-3-large")

    # Initialize the retriever
    retriever = VectorRetriever(
        driver,
        index_name=POSTER_INDEX_NAME,
        embedder=embedder,
    )

    query_text = "How do I do similarity search in Neo4j?"
    retriever_result = retriever.search(query_text=query_text, top_k=3)


Embedders
-----------------------------

Currently, we support two embedders: `OpenAIEmbeddings` and `SentenceTransformerEmbeddings`.

The OpenAIEmbedder was illustrated above. Here is how to use the `SentenceTransformerEmbeddings`:

.. code:: python

    from neo4j_genai.embeddings import SentenceTransformerEmbeddings

    embedder = SentenceTransformerEmbeddings(model="all-MiniLM-L6-v2")  # Note: this is the default model


If you want to use another embedder, you can create your own custom embedder. For instance,
this is an implementation of an embedder that would return only random numbers:

.. warning::
    Do not use it in your application :)


.. code:: python

    import random
    from neo4j_genai.embedder import Embedder

    class RandomEmbedder(Embedder):
        def __init__(
            self,
            size: int,
            seed: int = 42,
        ) -> None:
            self.size = size
            random.seed(seed)

        def embed_query(self, text: str) -> list[float]:
            return [
                random.random()
                for _ in range(self.size)
            ]

    embedder = RandomEmbedder(10)
    vector = embedder.embed_query("some text")


Other vector retriever configuration
----------------------------------------

Often, you won't be interested in all node properties, only a few of them will be
relevant to be added to the context in the LLM prompt. You can configure the properties
to be returned using the `return_properties` parameter:

.. code:: python

    from neo4j_genai.retrievers import VectorRetriever

    retriever = VectorRetriever(
        driver,
        index_name=POSTER_INDEX_NAME,
        embedder=embedder,
        return_properties=["title"],
    )


Use pre-filters
-----------------------------

When performing a similarity search, you may have constraints that you want to apply.
For instance, you may want to filter out movies released before 2000. This can be
achieved by specifying `filters`

.. code:: python

    from neo4j_genai.retrievers import VectorRetriever

    retriever = VectorRetriever(
        driver,
        index_name=POSTER_INDEX_NAME,
    )

    filters = {
        "year": {
            "$gte": 2000,
        }
    }

    query_text = "How do I do similarity search in Neo4j?"
    retriever_result = retriever.search(query_text=query_text, filters=filters)

.. note::

    Filters are implemented for all retrievers except the Hybrid retrievers.


The currently supported operators are:

- `$eq`: equal.
- `$ne`: not equal.
- `$lt`: less than.
- `$lte`: less than or equal to.
- `$gt`: greater than.
- `$gte`: greater than or equal to.
- `$between`: between.
- `$in`: value is in a given list.
- `$nin`: not in.
- `$like`: LIKE operator case-sensitive.
- `$ilike`: LIKE operator case-insensitive.


Here are example of valid filters and their meaning:

.. list-table:: Filters syntax
   :widths: 80 80
   :header-rows: 1

   * - Filter
     - Meaning
   * - {"year": 1999}
     - year = 1999
   * - {"year": {"$eq": 1999}}
     - year = 1999
   * - {"year": 2000, "title": "The Matrix"}
     - year = 1999 AND title = "The Matrix"
   * - {"$and": [{"year": 2000}, {"title": "The Matrix"}]}
     - year = 1999 AND title = "The Matrix"
   * - {"$or": [{"title": "The Matrix Revolution"}, {"title": "The Matrix"}]}
     - title = "The Matrix" OR title = "The Matrix Revolution"
   * - {"title": {"$like": "The Matrix"}}
     - title CONTAINS "The Matrix"
   * - {"title": {"$ilike": "the matrix"}}
     - toLower(title) CONTAINS "The Matrix"


Also see :ref:`vectorretriever`.


Vector Cypher retriever
=======================

The `VectorCypherRetriever` lets you take full advantage of the graph nature of Neo4j, by enhancing the
context with graph traversal.

Retrieval query
-----------------------------

To write the retrieval query, you must know that two variables are available in the query scope:

- `node`: the node returned by the vector index search
- `score`: the similarity score

Assuming we are using a graph of movies with actors the vector index is on some movie
properties, we can write the following retrieval query:

.. code:: python

    retriever = VectorCypherRetriever(
        driver,
        index_name=INDEX_NAME,
        retrieval_query="MATCH (node)<-[:ACTED_IN]-(p:Person) RETURN node.title as movieTitle, node.plot as movieDescription, collect(p.name) as actors, score",
    )


Format the results
-----------------------------

.. warning::

    This API is in beta mode and will be subject to change is the future.

For both readability and convenience for prompt-engineering, you have the ability to
format the result according to your needs by providing a record_formatter function to
the cypher retrievers. This function takes the neo4j record returned by the retrieval query
and must return a `RetrieverResultItem` with content (str) and metadata (dict) fields.
The content is the one that will be passed to the LLM, metadata can be used for debugging
purposes for instance.


.. code:: python

    def result_formatter(record: neo4j.Record) -> RetrieverResultItem:
        return RetrieverResultItem(
            content=f"Movie title: {record.get('movieTitle')}, description: {record.get('movieDescription')}, actors: {record.get('actors')}",
            metadata={
                "title": record.get('movieTitle'),
                "score": record.get("score"),
            }
        )

    retriever = VectorCypherRetriever(
        driver,
        index_name=INDEX_NAME,
        retrieval_query="MATCH (node)<-[:ACTED_IN]-(p:Person) RETURN node.title as movieTitle, node.plot as movieDescription, collect(p.name) as actors, score",
        result_formatter=result_formatter,
    )

Also see :ref:`vectorcypherretriever`.

Vector Databases
====================

.. note::

    For external retrievers, the filter syntax depends on the provider. Please refer to
    the documentation of the Python client for each provider for details.


Weaviate retrievers
-------------------

.. note::

    In order to import this retriever, you must install the Weaviate Python client:
    `pip install weaviate-client`


.. code:: python

    from weaviate.connect.helpers import connect_to_local
    from neo4j_genai.retrievers import WeaviateNeo4jRetriever

    client = connect_to_local()
    retriever = WeaviateNeo4jRetriever(
        driver=driver,
        client=client,
        embedder=embeder,
        collection="Movies",
        id_property_external="neo4j_id",
        id_property_neo4j="id",
    )

Internally, this retriever performs the vector search in Weaviate, finds the corresponding node by matching
the Weaviate metadata `id_property_external` with a Neo4j `node.id_property_neo4j`, and returns the matched node.

Similarly to the vector retriever, you can also use `return_properties` or `retrieval_query` parameters.

Also see :ref:`weaviateneo4jretriever`.

Pinecone retrievers
-------------------

.. note::

    In order to import this retriever, you must install the Weaviate Python client:
    `pip install pinecone-client`


.. code:: python

    from pinecone import Pinecone
    from neo4j_genai.retrievers import PineconeNeo4jRetriever

    client = Pinecone()  # ... create your Pinecone client

    retriever = PineconeNeo4jRetriever(
        driver=driver,
        client=client,
        index_name="Movies",
        id_property_neo4j="id",
        embedder=embeder,
    )

Also see :ref:`pineconeneo4jretriever`.


Other retrievers
===================

Hybrid and Hybrid Cypher retrievers
------------------------------------

See :ref:`hybridretriever` and :ref:`hybridcypherretriever`.


Custom retriever
===================

If none of the above matches your needs in terms of retrieval, you can implement your own custom retriever:

.. code:: python

    from neo4j_genai.retrievers.base import Retriever

    class VectorRetriever(Retriever):
        def __init__(
            self,
            driver: neo4j.Driver,
            # any other required parameters
        ) -> None:
            super().__init__(driver)

        def get_search_results(
            self,
            query_vector: Optional[list[float]] = None,
            query_text: Optional[str] = None,
            top_k: int = 5,
            filters: Optional[dict[str, Any]] = None,
        ) -> RawSearchResult:
            pass


See :ref:`rawsearchresult` for a description of the returned type.


******************************
DB operations
******************************

Create the vector index
========================

.. code:: python

    from neo4j import GraphDatabase
    from neo4j_genai.indexes import create_vector_index

    URI = "neo4j://localhost:7687"
    AUTH = ("neo4j", "password")

    INDEX_NAME = "chunk-index"
    DIMENSION=1536

    # Connect to Neo4j database
    driver = GraphDatabase.driver(URI, auth=AUTH)

    # Creating the index
    create_vector_index(
        driver,
        INDEX_NAME,
        label="Document",
        embedding_property="vectorProperty",
        dimensions=DIMENSION,
        similarity_fn="euclidean",
    )


Populate the vector index
==========================

.. code:: python

    from neo4j import GraphDatabase
    from random import random

    URI = "neo4j://localhost:7687"
    AUTH = ("neo4j", "password")

    # Connect to Neo4j database
    driver = GraphDatabase.driver(URI, auth=AUTH)

    # Upsert the vector
    vector = [random() for _ in range(DIMENSION)]
    upsert_vector(driver, node_id="1234", embedding_property="embedding", vector=vector)

This will update the node with `id(node)=1234` to add (or update) a `node.embedding` property.
This property will also be added to the vector index.


Drop the vector index
========================

.. warning::

    This operation can not be undone, use it with caution.


.. code:: python

    from neo4j import GraphDatabase

    URI = "neo4j://localhost:7687"
    AUTH = ("neo4j", "password")

    # Connect to Neo4j database
    driver = GraphDatabase.driver(URI, auth=AUTH)
    drop_index_if_exists(driver, INDEX_NAME)
