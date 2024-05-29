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


******
Errors
******


RetrieverInitializationError
============================

.. autoclass:: neo4j_genai.exceptions.RetrieverInitializationError
   :members:


SearchValidationError
=====================

.. autoclass:: neo4j_genai.exceptions.SearchValidationError
   :members:


FilterValidationError
=====================

.. autoclass:: neo4j_genai.exceptions.FilterValidationError
   :members:


EmbeddingRequiredError
======================

.. autoclass:: neo4j_genai.exceptions.EmbeddingRequiredError
   :members:


RecordCreationError
===================

.. autoclass:: neo4j_genai.exceptions.RecordCreationError
   :members:


Neo4jIndexError
===============

.. autoclass:: neo4j_genai.exceptions.Neo4jIndexError
   :members:


Neo4jVersionError
=================

.. autoclass:: neo4j_genai.exceptions.Neo4jVersionError
   :members:
