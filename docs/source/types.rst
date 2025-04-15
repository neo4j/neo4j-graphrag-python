.. _types-documentation:

*****
Types
*****

RawSearchResult
===============

.. autoclass:: neo4j_graphrag.types.RawSearchResult


RetrieverResult
===============

.. autoclass:: neo4j_graphrag.types.RetrieverResult


RetrieverResultItem
===================

.. autoclass:: neo4j_graphrag.types.RetrieverResultItem


LLMResponse
===========

.. autoclass:: neo4j_graphrag.llm.types.LLMResponse


LLMMessage
===========

.. autoclass:: neo4j_graphrag.types.LLMMessage


RagResultModel
==============

.. autoclass:: neo4j_graphrag.generation.types.RagResultModel

DocumentInfo
============

.. autoclass:: neo4j_graphrag.experimental.components.types.DocumentInfo


TextChunk
=========

.. autoclass:: neo4j_graphrag.experimental.components.types.TextChunk

TextChunks
==========

.. autoclass:: neo4j_graphrag.experimental.components.types.TextChunks

Neo4jNode
=========

.. autoclass:: neo4j_graphrag.experimental.components.types.Neo4jNode

Neo4jRelationship
=================

.. autoclass:: neo4j_graphrag.experimental.components.types.Neo4jRelationship

Neo4jGraph
==========

.. autoclass:: neo4j_graphrag.experimental.components.types.Neo4jGraph

KGWriterModel
=============

.. autoclass:: neo4j_graphrag.experimental.components.kg_writer.KGWriterModel

SchemaProperty
==============

.. autoclass:: neo4j_graphrag.experimental.components.schema.SchemaProperty

SchemaEntity
============

.. autoclass:: neo4j_graphrag.experimental.components.schema.SchemaEntity

SchemaRelation
==============

.. autoclass:: neo4j_graphrag.experimental.components.schema.SchemaRelation

SchemaConfig
============

.. autoclass:: neo4j_graphrag.experimental.components.schema.SchemaConfig

LexicalGraphConfig
===================

.. autoclass:: neo4j_graphrag.experimental.components.types.LexicalGraphConfig


Neo4jDriverType
===============

.. autoclass:: neo4j_graphrag.experimental.pipeline.config.object_config.Neo4jDriverType


Neo4jDriverConfig
=================

.. autoclass:: neo4j_graphrag.experimental.pipeline.config.object_config.Neo4jDriverConfig


LLMType
=======

.. autoclass:: neo4j_graphrag.experimental.pipeline.config.object_config.LLMType


LLMConfig
=========

.. autoclass:: neo4j_graphrag.experimental.pipeline.config.object_config.LLMConfig


EmbedderType
============

.. autoclass:: neo4j_graphrag.experimental.pipeline.config.object_config.EmbedderType


EmbedderConfig
==============

.. autoclass:: neo4j_graphrag.experimental.pipeline.config.object_config.EmbedderConfig


ComponentType
=============

.. autoclass:: neo4j_graphrag.experimental.pipeline.config.object_config.ComponentType


ComponentConfig
===============

.. autoclass:: neo4j_graphrag.experimental.pipeline.config.object_config.ComponentConfig


ParamFromEnvConfig
==================

.. autoclass:: neo4j_graphrag.experimental.pipeline.config.param_resolver.ParamFromEnvConfig


EventType
=========

.. autoenum:: neo4j_graphrag.experimental.pipeline.notification.EventType


PipelineEvent
==============

.. autoclass:: neo4j_graphrag.experimental.pipeline.notification.PipelineEvent

TaskEvent
==============

.. autoclass:: neo4j_graphrag.experimental.pipeline.notification.TaskEvent


EventCallbackProtocol
=====================

.. autoclass:: neo4j_graphrag.experimental.pipeline.notification.EventCallbackProtocol
    :members: __call__


TaskProgressCallbackProtocol
============================

.. autoclass:: neo4j_graphrag.experimental.pipeline.types.context.TaskProgressNotifierProtocol
    :members: __call__


RunContext
==========

.. autoclass:: neo4j_graphrag.experimental.pipeline.types.context.RunContext
