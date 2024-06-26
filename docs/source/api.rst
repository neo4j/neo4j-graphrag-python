.. _api-documentation:

API Documentation
#################

.. _retrievers-section:

**********
Retrievers
**********

RetrieverInterface
===================

.. autoclass:: neo4j_genai.retrievers.base.Retriever
    :members:


VectorRetriever
===============

.. autoclass:: neo4j_genai.retrievers.vector.VectorRetriever
    :members: search

VectorCypherRetriever
=====================

.. autoclass:: neo4j_genai.retrievers.vector.VectorCypherRetriever
    :members: search


HybridRetriever
===============

.. autoclass:: neo4j_genai.retrievers.hybrid.HybridRetriever
    :members: search


HybridCypherRetriever
=====================

.. autoclass:: neo4j_genai.retrievers.hybrid.HybridCypherRetriever
    :members: search



*******************
External Retrievers
*******************

This section includes retrievers that integrate with databases external to Neo4j.


WeaviateNeo4jRetriever
======================

.. autoclass:: neo4j_genai.retrievers.external.weaviate.weaviate.WeaviateNeo4jRetriever
    :members: search


PineconeNeo4jRetriever
======================

.. autoclass:: neo4j_genai.retrievers.external.pinecone.pinecone.PineconeNeo4jRetriever
    :members: search


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

  * :class:`neo4j_genai.exceptions.RagInitializationError`

  * :class:`neo4j_genai.exceptions.PromptMissingInputError`

  * :class:`neo4j_genai.exceptions.LLMGenerationError`


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
===================

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


RagInitializationError
==========================

.. autoclass:: neo4j_genai.exceptions.RagInitializationError
   :show-inheritance:


PromptMissingInputError
==========================

.. autoclass:: neo4j_genai.exceptions.PromptMissingInputError
   :show-inheritance:


LLMGenerationError
==========================

.. autoclass:: neo4j_genai.exceptions.LLMGenerationError
   :show-inheritance:
