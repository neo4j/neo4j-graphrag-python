.. _api-documentation:

API Documentation
#################

.. _components-section:

**********
Components
**********

KGWriter
========

.. autoclass:: neo4j_genai.components.kg_writer.KGWriter
    :members: run

Neo4jWriter
===========

.. autoclass:: neo4j_genai.components.kg_writer.Neo4jWriter
    :members: run

TextSplitter
============

.. autoclass:: neo4j_genai.components.text_splitters.base.TextSplitter
    :members: run

LangChainTextSplitterAdapter
============================

.. autoclass:: neo4j_genai.components.text_splitters.langchain.LangChainTextSplitterAdapter
    :members: run

LlamaIndexTextSplitterAdapter
=============================

.. autoclass:: neo4j_genai.components.text_splitters.llamaindex.LlamaIndexTextSplitterAdapter
    :members: run

TextChunkEmbedder
=================

.. autoclass:: neo4j_genai.components.embedder.TextChunkEmbedder
    :members: run

SchemaBuilder
=============

.. autoclass:: neo4j_genai.components.schema.SchemaBuilder
    :members: run

EntityRelationExtractor
=======================

.. autoclass:: neo4j_genai.components.entity_relation_extractor.EntityRelationExtractor
    :members: run

LLMEntityRelationExtractor
==========================

.. autoclass:: neo4j_genai.components.entity_relation_extractor.LLMEntityRelationExtractor
    :members: run

.. _retrievers-section:

**********
Retrievers
**********

RetrieverInterface
==================

.. autoclass:: neo4j_genai.retrievers.base.Retriever
    :members:


VectorRetriever
===============

.. autoclass:: neo4j_genai.retrievers.VectorRetriever
    :members: search

VectorCypherRetriever
=====================

.. autoclass:: neo4j_genai.retrievers.VectorCypherRetriever
    :members: search


HybridRetriever
===============

.. autoclass:: neo4j_genai.retrievers.HybridRetriever
    :members: search


HybridCypherRetriever
=====================

.. autoclass:: neo4j_genai.retrievers.HybridCypherRetriever
    :members: search

Text2CypherRetriever
=====================

.. autoclass:: neo4j_genai.retrievers.Text2CypherRetriever
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


********
Embedder
********

.. autoclass:: neo4j_genai.embedder.Embedder
    :members:

SentenceTransformerEmbeddings
================================

.. autoclass:: neo4j_genai.embeddings.sentence_transformers.SentenceTransformerEmbeddings
    :members:

**********
Generation
**********

LLMInterface
============

.. autoclass:: neo4j_genai.llm.LLMInterface
    :members:


OpenAILLM
=========

.. autoclass:: neo4j_genai.llm.OpenAILLM
    :members:


PromptTemplate
==============

.. autoclass:: neo4j_genai.generation.prompts.PromptTemplate
    :members:

.. _database-interaction-section:

********************
Database Interaction
********************

.. _create-vector-index:

.. autofunction:: neo4j_genai.indexes.create_vector_index

.. _create-fulltext-index:

.. autofunction:: neo4j_genai.indexes.create_fulltext_index

.. autofunction:: neo4j_genai.indexes.drop_index_if_exists

.. autofunction:: neo4j_genai.indexes.upsert_vector

.. autofunction:: neo4j_genai.indexes.upsert_vector_on_relationship


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

  * :class:`neo4j_genai.pipeline.exceptions.PipelineDefinitionError`

  * :class:`neo4j_genai.pipeline.exceptions.PipelineMissingDependencyError`

  * :class:`neo4j_genai.pipeline.exceptions.PipelineStatusUpdateError`


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
=========================

.. autoclass:: neo4j_genai.exceptions.Text2CypherRetrievalError
   :show-inheritance:


SchemaFetchError
================

.. autoclass:: neo4j_genai.exceptions.SchemaFetchError
   :show-inheritance:


RagInitializationError
======================

.. autoclass:: neo4j_genai.exceptions.RagInitializationError
   :show-inheritance:


PromptMissingInputError
=======================

.. autoclass:: neo4j_genai.exceptions.PromptMissingInputError
   :show-inheritance:


LLMGenerationError
==================

.. autoclass:: neo4j_genai.exceptions.LLMGenerationError
   :show-inheritance:


PipelineDefinitionError
=======================

.. autoclass:: neo4j_genai.pipeline.exceptions.PipelineDefinitionError
   :show-inheritance:


PipelineMissingDependencyError
==============================

.. autoclass:: neo4j_genai.pipeline.exceptions.PipelineMissingDependencyError
   :show-inheritance:


PipelineStatusUpdateError
=========================

.. autoclass:: neo4j_genai.pipeline.exceptions.PipelineStatusUpdateError
   :show-inheritance:
