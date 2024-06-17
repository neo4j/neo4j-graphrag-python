.. _api-documentation:

API Documentation
#################

************************************
Retrieval-Augmented Generation (RAG)
************************************
RAG is a technique that enhances Large Language Model (LLM) responses by retrieving
source information from external data stores to augment generated responses.

This package enables Python developers to perform RAG using Neo4j.

**********
Retrievers
**********

VectorRetriever
===============

.. autoclass:: neo4j_genai.retrievers.vector.VectorRetriever
    :members:

VectorCypherRetriever
=====================

.. autoclass:: neo4j_genai.retrievers.vector.VectorCypherRetriever
   :members:


HybridRetriever
===============

.. autoclass:: neo4j_genai.retrievers.hybrid.HybridRetriever
   :members:


HybridCypherRetriever
=====================

.. autoclass:: neo4j_genai.retrievers.hybrid.HybridCypherRetriever
   :members:



*******************
External Retrievers
*******************

This section includes retrievers that integrate with databases external to Neo4j.


WeaviateNeo4jRetriever
======================

.. autoclass:: neo4j_genai.retrievers.external.weaviate.WeaviateNeo4jRetriever
   :members:


PineconeNeo4jRetriever
======================

.. autoclass:: neo4j_genai.retrievers.external.pinecone.PineconeNeo4jRetriever
   :members:


******
Errors
******


* :class:`neo4j_genai.exceptions.Neo4jGenAiError`

  * :class:`neo4j_genai.exceptions.RetrieverInitializationError`

  * :class:`neo4j_genai.exceptions.SearchValidationError`

  * :class:`neo4j_genai.exceptions.FilterValidationError`

  * :class:`neo4j_genai.exceptions.EmbeddingRequiredError`

  * :class:`neo4j_genai.exceptions.InvalidRetrieverResultError`

  * :class:`neo4j_genai.exceptions.Neo4jIndexError`

  * :class:`neo4j_genai.exceptions.Neo4jVersionError`

  * :class:`neo4j_genai.exceptions.Text2CypherRetrievalError`

  * :class:`neo4j_genai.exceptions.SchemaFetchError`


Neo4jGenAiError
===============

.. autoclass:: neo4j_genai.exceptions.Neo4jGenAiError
   :show-inheritance:


RetrieverInitializationError
============================

.. autoclass:: neo4j_genai.exceptions.RetrieverInitializationError
   :show-inheritance:


SearchValidationError
=====================

.. autoclass:: neo4j_genai.exceptions.SearchValidationError
   :show-inheritance:


FilterValidationError
=====================

.. autoclass:: neo4j_genai.exceptions.FilterValidationError
   :show-inheritance:


EmbeddingRequiredError
======================

.. autoclass:: neo4j_genai.exceptions.EmbeddingRequiredError
   :show-inheritance:


InvalidRetrieverResultError
===========================

.. autoclass:: neo4j_genai.exceptions.InvalidRetrieverResultError
   :show-inheritance:


Neo4jIndexError
===============

.. autoclass:: neo4j_genai.exceptions.Neo4jIndexError
   :show-inheritance:


Neo4jInsertionError
===============

.. autoclass:: neo4j_genai.exceptions.Neo4jInsertionError
   :show-inheritance:


Neo4jVersionError
=================

.. autoclass:: neo4j_genai.exceptions.Neo4jVersionError
   :show-inheritance:


Text2CypherRetrievalError
==========================

.. autoclass:: neo4j_genai.exceptions.Text2CypherRetrievalError
   :show-inheritance:


SchemaFetchError
================

.. autoclass:: neo4j_genai.exceptions.SchemaFetchError
   :show-inheritance:
