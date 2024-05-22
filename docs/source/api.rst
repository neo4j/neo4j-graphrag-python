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
