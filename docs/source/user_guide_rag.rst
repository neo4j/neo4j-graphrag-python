.. _user-guide-rag:

User Guide: RAG
#################

This guide provides a starting point for using the Neo4j GraphRAG package
and configuring it according to specific requirements.


************
Quickstart
************

To perform a GraphRAG query using the `neo4j-graphrag` package, a few components are needed:

1. A Neo4j driver: used to query your Neo4j database.
2. A Retriever: the `neo4j-graphrag` package provides some implementations (see the :ref:`dedicated section <retriever-configuration>`) and lets you write your own if none of the provided implementations matches your needs (see :ref:`how to write a custom retriever <custom-retriever>`).
3. An LLM: to generate the answer, we need to call an LLM model. The neo4j-graphrag package's LLM interface is compatible with LangChain. Developers can also write their own interface if needed.

In practice, it's done with only a few lines of code:

.. code:: python

    from neo4j import GraphDatabase
    from neo4j_graphrag.retrievers import VectorRetriever
    from neo4j_graphrag.llm import OpenAILLM
    from neo4j_graphrag.generation import GraphRAG
    from neo4j_graphrag.embeddings import OpenAIEmbeddings

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


.. warning::

    Using `OpenAILLM` requires the `openai` Python client. You can install it with `pip install "neo4j_graphrag[openai]"`.


The following sections provide more details about how to customize this code.

******************************
GraphRAG Configuration
******************************

Each component can be configured individually: the LLM and the prompt.

Using Another LLM Model
========================

If OpenAI cannot be used directly, there are a few available alternatives:

- Use Azure OpenAI (GPT...).
- Use Google VertexAI (Gemini...).
- Use Anthropic LLM (Claude...).
- Use Mistral LLM
- Use Cohere.
- Use a local Ollama model.
- Implement a custom interface.
- Utilize any LangChain chat model.

All options are illustrated below.

Using Azure Open AI LLM
-----------------------

It is possible to use Azure OpenAI switching to the `AzureOpenAILLM` class:

.. code:: python

    from neo4j_graphrag.llm import AzureOpenAILLM
    llm = AzureOpenAILLM(
        model_name="gpt-4o",
        azure_endpoint="https://example-endpoint.openai.azure.com/",  # update with your endpoint
        api_version="2024-06-01",  # update appropriate version
        api_key="...",  # api_key is optional and can also be set with OPENAI_API_KEY env var
    )
    llm.invoke("say something")

Check the OpenAI Python client `documentation <https://github.com/openai/openai-python?tab=readme-ov-file#microsoft-azure-openai>`_.
to learn more about the configuration.

.. note::

    In order to run this code, the `openai` Python package needs to be installed:
    `pip install "neo4j_graphrag[openai]"`


See :ref:`azureopenaillm`.


Using VertexAI LLM
------------------

To use VertexAI, instantiate the `VertexAILLM` class:

.. code:: python

    from neo4j_graphrag.llm import VertexAILLM
    from vertexai.generative_models import GenerationConfig

    generation_config = GenerationConfig(temperature=0.0)
    llm = VertexAILLM(
        model_name="gemini-1.5-flash-001", generation_config=generation_config
    )
    llm.invoke("say something")


.. note::

    In order to run this code, the `google-cloud-aiplatform` Python package needs to be installed:
    `pip install "neo4j_grpahrag[vertexai]"`


See :ref:`vertexaillm`.


Using Anthropic LLM
-------------------

To use Anthropic, instantiate the `AnthropicLLM` class:

.. code:: python

    from neo4j_graphrag.llm import AnthropicLLM

    llm = AnthropicLLM(
        model_name="claude-3-opus-20240229",
        model_params={"max_tokens": 1000},  # max_tokens must be specified
        api_key=api_key,  # can also set `ANTHROPIC_API_KEY` in env vars
    )
    llm.invoke("say something")


.. note::

    In order to run this code, the `anthropic` Python package needs to be installed:
    `pip install "neo4j_graphrag[anthropic]"`

See :ref:`anthropicllm`.


Using MistralAI LLM
-------------------

To use MistralAI, instantiate the `MistralAILLM` class:

.. code:: python

    from neo4j_graphrag.llm import MistralAILLM

    llm = MistralAILLM(
        model_name="mistral-small-latest",
        api_key=api_key,  # can also set `MISTRAL_API_KEY` in env vars
    )
    llm.invoke("say something")


.. note::

    In order to run this code, the `mistralai` Python package needs to be installed:
    `pip install "neo4j_graphrag[mistralai]"`

See :ref:`mistralaillm`.



Using Cohere LLM
----------------

To use Cohere, instantiate the `CohereLLM` class:

.. code:: python

    from neo4j_graphrag.llm import CohereLLM

    llm = CohereLLM(
        model_name="command-r",
        api_key=api_key,  # can also set `CO_API_KEY` in env vars
    )
    llm.invoke("say something")


.. note::

    In order to run this code, the `cohere` Python package needs to be installed:
    `pip install "neo4j_graphrag[cohere]"`


See :ref:`coherellm`.


Using a Local Model via Ollama
-------------------------------

Assuming Ollama is running on the default address `127.0.0.1:11434`,
it can be queried using the following:

.. code:: python

    from neo4j_graphrag.llm import OllamaLLM
    llm = OllamaLLM(model_name="orca-mini")
    llm.invoke("say something")


Using a Model from LangChain
-----------------------------

The LangChain Python package contains implementations for various LLMs and providers.
Its interface is compatible with our `GraphRAG` interface, facilitating integration:

.. code:: python

    from neo4j_graphrag.generation import GraphRAG
    from langchain_community.chat_models import ChatOllama

    # retriever = ...

    llm = ChatOllama(model="llama3:8b")
    rag = GraphRAG(retriever=retriever, llm=llm)
    query_text = "How do I do similarity search in Neo4j?"
    response = rag.search(query_text=query_text, retriever_config={"top_k": 5})
    print(response.answer)


It is however not mandatory to use LangChain.

Using a Custom Model
--------------------

If the provided implementations do not match their needs, developers can create a
custom LLM class by subclassing the `LLMInterface`.
Here's an example using the Python Ollama client:


.. code:: python

    import ollama
    from neo4j_graphrag.llm import LLMInterface, LLMResponse

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

        async def ainvoke(self, input: str) -> LLMResponse:
            return self.invoke(input)  # TODO: implement async with ollama.AsyncClient


    # retriever = ...

    llm = OllamaLLM("llama3:8b")

    rag = GraphRAG(retriever=retriever, llm=llm)
    query_text = "How do I do similarity search in Neo4j?"
    response = rag.search(query_text=query_text, retriever_config={"top_k": 5})
    print(response.answer)

See :ref:`llminterface`.


Configuring the Prompt
========================

Prompts are managed through `PromptTemplate` classes. Specifically, the `GraphRAG` pipeline
utilizes a `RagTemplate` with a default prompt that can be accessed through
`rag.prompt_template.template`. To use a different prompt, subclass the `RagTemplate`
class and pass it to the `GraphRAG` pipeline object during initialization:

.. code:: python

    from neo4j_graphrag.generation import RagTemplate, GraphRAG

    # retriever = ...
    # llm = ...

    prompt_template = RagTemplate(
        prompt="Answer the question {question} using context {context} and examples {examples}",
        expected_inputs=["context", "question", "examples"]
    )

    rag = GraphRAG(retriever=retriever, llm=llm, prompt_template=prompt_template)

    # ...


See :ref:`prompttemplate`.


The final configurable component in the `GraphRAG` pipeline is the retriever.
Below are descriptions of the various options available.

.. _retriever-configuration:

************************
Retriever Configuration
************************

We provide implementations for the following retrievers:

.. list-table:: List of retrievers
   :widths: 30 100
   :header-rows: 1

   * - Retriever
     - Description
   * - :ref:`VectorRetriever <vector-retriever-user-guide>`
     - Performs a similarity search based on a Neo4j vector index and a query text or vector. Returns the matched `node` and similarity `score`.
   * - :ref:`VectorCypherRetriever <vector-cypher-retriever-user-guide>`
     - Performs a similarity search based on a Neo4j vector index and a query text or vector. The returned results can be configured through a retrieval query parameter that will be executed after the index search. It can be used to fetch more context around the matched node.
   * - :ref:`HybridRetriever <hybrid-retriever-user-guide>`
     - Uses both a vector and a full-text index in Neo4j.
   * - :ref:`HybridCypherRetriever <hybrid-cypher-retriever-user-guide>`
     - Same as HybridRetriever with a retrieval query similar to VectorCypherRetriever.
   * - :ref:`Text2Cypher <text2cypher-retriever-user-guide>`
     - Translates the user question into a Cypher query to be run against a Neo4j database (or Knowledge Graph). The results of the query are then passed to the LLM to generate the final answer.
   * - :ref:`WeaviateNeo4jRetriever <weaviate-neo4j-retriever-user-guide>`
     - Use this retriever when vectors are saved in a Weaviate vector database
   * - :ref:`PineconeNeo4jRetriever <pinecone-neo4j-retriever-user-guide>`
     - Use this retriever when vectors are saved in a Pinecone vector database
   * - :ref:`QdrantNeo4jRetriever <qdrant-neo4j-retriever-user-guide>`
     - Use this retriever when vectors are saved in a Qdrant vector database

Retrievers all expose a `search` method that we will discuss in the next sections.


.. _vector-retriever-user-guide:

Vector Retriever
===================

The simplest method to instantiate a vector retriever is:

.. code:: python

    from neo4j_graphrag.retrievers import VectorRetriever

    retriever = VectorRetriever(
        driver,
        index_name=POSTER_INDEX_NAME,
    )

The `index_name` is the name of the Neo4j vector index that will be used for similarity search.


.. warning::

    Vector index use an **approximate nearest neighbor** algorithm.
    Refer to the `Neo4j Documentation <https://neo4j.com/docs/cypher-manual/current/indexes/semantic-indexes/vector-indexes/#limitations-and-issues>`_ to learn about its limitations.


Search Similar Vector
-----------------------------

To identify the top 3 most similar nodes, perform a search by vector:

.. code:: python

    vector = []  # a list of floats, same size as the vectors in the Neo4j vector index
    retriever_result = retriever.search(query_vector=vector, top_k=3)

However, in most cases, a text (from the user) will be provided instead of a vector.
In this scenario, an `Embedder` is required.

Search Similar Text
--------------------

When searching for a text, specifying how the retriever transforms (embeds) the text
into a vector is required. Therefore, the retriever requires knowledge of an embedder:

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
---------

Currently, this package supports the following embedders:

- :ref:`openaiembeddings`
- :ref:`sentencetransformerembeddings`
- :ref:`vertexaiembeddings`
- :ref:`mistralaiembeddings`
- :ref:`cohereembeddings`
- :ref:`azureopenaiembeddings`
- :ref:`ollamaembeddings`

The `OpenAIEmbeddings` was illustrated previously. Here is how to use the `SentenceTransformerEmbeddings`:

.. code:: python

    from neo4j_graphrag.embeddings import SentenceTransformerEmbeddings

    embedder = SentenceTransformerEmbeddings(model="all-MiniLM-L6-v2")  # Note: this is the default model


If another embedder is desired, a custom embedder can be created, using the `Embedder` interface.


Other Vector Retriever Configuration
----------------------------------------

Often, not all node properties are pertinent for the RAG context; only a selected few are relevant
for inclusion in the LLM prompt context. You can specify which properties to return
using the `return_properties` parameter:

.. code:: python

    from neo4j_graphrag.retrievers import VectorRetriever

    retriever = VectorRetriever(
        driver,
        index_name=POSTER_INDEX_NAME,
        embedder=embedder,
        return_properties=["title"],
    )


Pre-Filters
-----------

When performing a similarity search, one may have constraints to apply.
For instance, filtering out movies released before 2000. This can be achieved
using `filters`.

.. note::

    Filters are implemented for all retrievers except the Hybrid retrievers.
    The documentation below is not valid for external retrievers, which use
    their own filter syntax (see :ref:`vector-databases-section`).


.. code:: python

    from neo4j_graphrag.retrievers import VectorRetriever

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

.. warning::

    When using filters, the similarity search bypasses the vector index and instead utilizes
    an exact match algorithm
    Ensure that the pre-filtering is stringent enough to prevent query overload.

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


Here are examples of valid filter syntaxes and their meaning:

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


See also :ref:`vectorretriever`.

.. _vector-cypher-retriever-user-guide:

Vector Cypher Retriever
=======================

The `VectorCypherRetriever` fully leverages Neo4j's graph capabilities by combining vector-based similarity searches with graph traversal techniques. It processes a query embedding to perform a similarity search against a specified vector index, retrieves relevant node variables, and then executes a Cypher query to traverse the graph based on these nodes. This integration ensures that retrievals are both semantically meaningful and contextually enriched by the underlying graph structure.


Retrieval Query
---------------

When crafting the retrieval query, it's important to note two available variables
are in the query scope:

- `node`: represents the node retrieved from the vector index search.
- `score`: denotes the similarity score.

For instance, in a movie graph with actors where the vector index pertains to
certain movie properties, the retrieval query can be structured as follows:

.. code:: python

    retrieval_query = """
    RETURN  node.title as movieTitle,
            node.plot as moviePlot,
            collect { MATCH (actor:Actor)-[:ACTED_IN]->(node) RETURN a.name } AS actors
    """
    retriever = VectorCypherRetriever(
        driver,
        index_name=INDEX_NAME,
        retrieval_query=retrieval_query,
    )


It is recommended that the retrieval query returns node properties, as opposed to nodes.


Format the Results
------------------

.. warning::

    This API is in beta mode and will be subject to change in the future.

The result_formatter function customizes the output of Cypher retrievers for improved prompt engineering and readability. It converts each Neo4j record into a RetrieverResultItem with two fields: `content` and `metadata`.

The `content` field is a formatted string containing the key information intended for the language model, such as movie titles or descriptions. The `metadata` field holds additional details, useful for debugging or providing extra context, like scores or node properties.


.. code:: python

    def result_formatter(record: neo4j.Record) -> RetrieverResultItem:
        content=f"Movie title: {record.get('movieTitle')}, description: {record.get('movieDescription')}, actors: {record.get('actors')}",
        return RetrieverResultItem(
            metadata={
                "title": record.get('movieTitle'),
                "score": record.get("score"),
            }
        )

    retriever = VectorCypherRetriever(
        driver,
        index_name=INDEX_NAME,
        retrieval_query="OPTIONAL MATCH (node)<-[:ACTED_IN]-(p:Person) RETURN node.title as movieTitle, node.plot as movieDescription, collect(p.name) as actors, score",
        result_formatter=result_formatter,
    )

Also see :ref:`vectorcypherretriever`.


.. _vector-databases-section:

Vector Databases
====================

.. note::

    For external retrievers, the filter syntax depends on the provider. Please refer to
    the documentation of the Python client for each provider for details.

.. _weaviate-neo4j-retriever-user-guide:

Weaviate Retrievers
-------------------

.. note::

    In order to import this retriever, the Weaviate Python client must be installed:
    `pip install "neo4j_graphrag[weaviate]"`


.. code:: python

    from weaviate.connect.helpers import connect_to_local
    from neo4j_graphrag.retrievers import WeaviateNeo4jRetriever

    client = connect_to_local()
    retriever = WeaviateNeo4jRetriever(
        driver=driver,
        client=client,
        embedder=embedder,
        collection="Movies",
        id_property_external="neo4j_id",
        id_property_neo4j="id",
    )

Internally, this retriever performs the vector search in Weaviate, finds the corresponding node by matching
the Weaviate metadata `id_property_external` with a Neo4j `node.id_property_neo4j`, and returns the matched node.

The `return_properties` and `retrieval_query` parameters operate similarly to those in other retrievers.

See :ref:`weaviateneo4jretriever`.

.. _pinecone-neo4j-retriever-user-guide:

Pinecone Retrievers
-------------------

.. note::

    In order to import this retriever, the Pinecone Python client must be installed:
    `pip install "neo4j_graphrag[pinecone]"`


.. code:: python

    from pinecone import Pinecone
    from neo4j_graphrag.retrievers import PineconeNeo4jRetriever

    client = Pinecone()  # ... create your Pinecone client

    retriever = PineconeNeo4jRetriever(
        driver=driver,
        client=client,
        index_name="Movies",
        id_property_neo4j="id",
        embedder=embedder,
    )

Also see :ref:`pineconeneo4jretriever`.

.. _qdrant-neo4j-retriever-user-guide:

Qdrant Retrievers
-----------------

.. note::

    In order to import this retriever, the Qdrant Python client must be installed:
    `pip install "neo4j_graphrag[qdrant]"`


.. code:: python

    from qdrant_client import QdrantClient
    from neo4j_graphrag.retrievers import QdrantNeo4jRetriever

    client = QdrantClient(...)  # construct the Qdrant client instance

    retriever = QdrantNeo4jRetriever(
        driver=driver,
        client=client,
        collection_name="my-collection",
        id_property_external="neo4j_id",    # The payload field that contains identifier to a corresponding Neo4j node id property
        id_property_neo4j="id",
        embedder=embedder,
    )

See :ref:`qdrantneo4jretriever`.


Other Retrievers
===================

.. _hybrid-retriever-user-guide:

Hybrid Retrievers
------------------------------------

In an hybrid retriever, results are searched for in both a vector and a full-text index.
For this reason, a full-text index must also exist in the database, and its name must
be provided when instantiating the retriever:

.. code:: python

    from neo4j_graphrag.retrievers import HybridRetriever

    INDEX_NAME = "embedding-name"
    FULLTEXT_INDEX_NAME = "fulltext-index-name"

    retriever = HybridRetriever(
        driver,
        INDEX_NAME,
        FULLTEXT_INDEX_NAME,
        embedder,
    )


See :ref:`hybridretriever`.

Also note that there is an helper function to create a full-text index  (see :ref:`the API documentation<create-fulltext-index>`).

.. _hybrid-cypher-retriever-user-guide:

Hybrid Cypher Retrievers
------------------------

In an hybrid cypher retriever, results are searched for in both a vector and a
full-text index. Once the similar nodes are identified, a retrieval query can traverse
the graph and return more context:

.. code:: python

    from neo4j_graphrag.retrievers import HybridCypherRetriever

    INDEX_NAME = "embedding-name"
    FULLTEXT_INDEX_NAME = "fulltext-index-name"

    retriever = HybridCypherRetriever(
        driver,
        INDEX_NAME,
        FULLTEXT_INDEX_NAME,
        retrieval_query="MATCH (node)-[:AUTHORED_BY]->(author:Author) RETURN author.name"
        embedder=embedder,
    )


See :ref:`hybridcypherretriever`.


.. _text2cypher-retriever-user-guide:

Text2Cypher Retriever
------------------------------------

This retriever first asks an LLM to generate a Cypher query to fetch the exact
information required to answer the question from the database. Then this query is
executed and the resulting records are added to the context for the LLM to write
the answer to the initial user question. The cypher-generation and answer-generation
LLMs can be different.

.. code:: python

    from neo4j import GraphDatabase
    from neo4j_graphrag.retrievers import Text2CypherRetriever
    from neo4j_graphrag.llm import OpenAILLM

    URI = "neo4j://localhost:7687"
    AUTH = ("neo4j", "password")

    # Connect to Neo4j database
    driver = GraphDatabase.driver(URI, auth=AUTH)

    # Create LLM object
    llm = OpenAILLM(model_name="gpt-4o")

    # (Optional) Specify your own Neo4j schema
    neo4j_schema = """
    Node properties:
    Person {name: STRING, born: INTEGER}
    Movie {tagline: STRING, title: STRING, released: INTEGER}
    Relationship properties:
    ACTED_IN {roles: LIST}
    REVIEWED {summary: STRING, rating: INTEGER}
    The relationships:
    (:Person)-[:ACTED_IN]->(:Movie)
    (:Person)-[:DIRECTED]->(:Movie)
    (:Person)-[:PRODUCED]->(:Movie)
    (:Person)-[:WROTE]->(:Movie)
    (:Person)-[:FOLLOWS]->(:Person)
    (:Person)-[:REVIEWED]->(:Movie)
    """

    # (Optional) Provide user input/query pairs for the LLM to use as examples
    examples = [
        "USER INPUT: 'Which actors starred in the Matrix?' QUERY: MATCH (p:Person)-[:ACTED_IN]->(m:Movie) WHERE m.title = 'The Matrix' RETURN p.name"
    ]

    # Initialize the retriever
    retriever = Text2CypherRetriever(
        driver=driver,
        llm=llm,  # type: ignore
        neo4j_schema=neo4j_schema,
        examples=examples,
    )

    # Generate a Cypher query using the LLM, send it to the Neo4j database, and return the results
    query_text = "Which movies did Hugo Weaving star in?"
    print(retriever.search(query_text=query_text))


.. warning::

    Using `OpenAILLM` requires the `openai` Python client. You can install it with `pip install "neo4j_graphrag[openai]"`.

.. note::

    Since we are not performing any similarity search (vector index), the Text2Cypher
    retriever does not require any embedder.

.. warning::

    The LLM-generated query is not guaranteed to be syntactically correct. In case it can't be
    executed, a `Text2CypherRetrievalError` is raised.


See :ref:`text2cypherretriever`.

.. _custom-retriever:

Custom Retriever
===================

If the application requires very specific retrieval strategy, it is possible to implement
a custom retriever using the `Retriever` interface:

.. code:: python

    from neo4j_graphrag.retrievers.base import Retriever

    class MyCustomRetriever(Retriever):
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
DB Operations
******************************

See :ref:`database-interaction-section`.

Create a Vector Index
========================

.. code:: python

    from neo4j import GraphDatabase
    from neo4j_graphrag.indexes import create_vector_index

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


Populate a Vector Index
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


Drop a Vector Index
========================

.. warning::

    This operation is irreversible and should be used with caution.


.. code:: python

    from neo4j import GraphDatabase

    URI = "neo4j://localhost:7687"
    AUTH = ("neo4j", "password")

    # Connect to Neo4j database
    driver = GraphDatabase.driver(URI, auth=AUTH)
    drop_index_if_exists(driver, INDEX_NAME)
