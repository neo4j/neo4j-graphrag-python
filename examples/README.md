# Examples Index

This folder contains examples usage for the different features
supported by the `neo4j-graphrag` package:

- [Build Knowledge Graph](#build-knowledge-graph) from PDF or text
- [Retrieve](#retrieve) information from the graph
- [Question Answering](#answer-graphrag) (Q&A)

Each of these steps have many customization options which
are listed in [the last section of this file](#customize).

## Build Knowledge Graph

- [End to end PDF to graph simple pipeline](build_graph/simple_kg_builder_from_pdf.py)
- [End to end text to graph simple pipeline](build_graph/simple_kg_builder_from_text.py)
- [Build KG pipeline from config file](build_graph/from_config_files/simple_kg_pipeline_from_config_file.py)
- [Build KG pipeline with PDF URL](build_graph/from_config_files/simple_kg_pipeline_from_config_file_with_url.py)


## Retrieve

- [Retriever from an embedding vector](retrieve/similarity_search_for_vector.py)
- [Retriever from a text](retrieve/similarity_search_for_text.py)
- [Graph-based retrieval with VectorCypherRetriever](retrieve/vector_cypher_retriever.py)
- [Hybrid retriever](./retrieve/hybrid_retriever.py)
- [Hybrid Cypher retriever](./retrieve/hybrid_cypher_retriever.py)
- [Text2Cypher retriever](./retrieve/text2cypher_search.py)


### External Retrievers

#### Weaviate

- [Vector search](customize/retrievers/external/weaviate/weaviate_vector_search.py)
- [Text search with local embeder](customize/retrievers/external/weaviate/weaviate_text_search_local_embedder.py)
- [Text search with remote embeder](customize/retrievers/external/weaviate/weaviate_text_search_remote_embedder.py)

#### Pinecone

- [Vector search](./customize/retrievers/external/pinecone/pinecone_vector_search.py)
- [Text search](./customize/retrievers/external/pinecone/pinecone_text_search.py)


### Qdrant

- [Vector search](./customize/retrievers/external/qdrant/qdrant_vector_search.py)
- [Text search](./customize/retrievers/external/qdrant/qdrant_text_search.py)


## Answer: GraphRAG

- [End to end GraphRAG](./answer/graphrag.py)
- [GraphRAG with message history](./question_answering/graphrag_with_message_history.py)


## Customize

### Retriever

- [Control result format for VectorRetriever](customize/retrievers/result_formatter_vector_retriever.py)
- [Control result format for VectorCypherRetriever](customize/retrievers/result_formatter_vector_cypher_retriever.py)
- [Use pre-filters](customize/retrievers/use_pre_filters.py)
- [Text2Cypher: use a custom prompt](customize/retrievers/text2cypher_custom_prompt.py)

### LLMs

- [OpenAI (GPT)](./customize/llms/openai_llm.py)
- [Azure OpenAI]()
- [VertexAI (Gemini)](./customize/llms/vertexai_llm.py)
- [MistralAI](./customize/llms/mistalai_llm.py)
- [Cohere](./customize/llms/cohere_llm.py)
- [Anthropic (Claude)](./customize/llms/anthropic_llm.py)
- [Ollama](./customize/llms/ollama_llm.py)
- [Custom LLM](./customize/llms/custom_llm.py)

- [Message history](./customize/llms/llm_with_message_history.py)
- [System Instruction](./customize/llms/llm_with_system_instructions.py)


### Prompts

- [Using a custom prompt for RAG](customize/answer/custom_prompt.py)


### Embedders

- [OpenAI](./customize/embeddings/openai_embeddings.py)
- [Azure OpenAI](./customize/embeddings/azure_openai_embeddings.py)
- [VertexAI](./customize/embeddings/vertexai_embeddings.py)
- [MistralAI](./customize/embeddings/mistalai_embeddings.py)
- [Cohere](./customize/embeddings/cohere_embeddings.py)
- [Ollama](./customize/embeddings/ollama_embeddings.py)
- [Custom LLM](./customize/embeddings/custom_embeddings.py)


### KG Construction - Pipeline

- [End to end example with explicit components and text input](./customize/build_graph/pipeline/kg_builder_from_text.py)
- [End to end example with explicit components and PDF input](./customize/build_graph/pipeline/kg_builder_from_pdf.py)
- [Process multiple documents](./customize/build_graph/pipeline/kg_builder_two_documents_entity_resolution.py)
- [Export lexical graph creation into another pipeline](./customize/build_graph/pipeline/text_to_lexical_graph_to_entity_graph_two_pipelines.py)
- [Build pipeline from config file](customize/build_graph/pipeline/from_config_files/pipeline_from_config_file.py)


#### Components

- Loaders:
  - [Load PDF file](./customize/build_graph/components/loaders/pdf_loader.py)
  - [Custom](./customize/build_graph/components/loaders/custom_loader.py)
- Text Splitter:
  - [Fixed size splitter](./customize/build_graph/components/splitters/fixed_size_splitter.py)
  - [Splitter from LangChain](./customize/build_graph/components/splitters/langhchain_splitter.py)
  - [Splitter from LLamaIndex](./customize/build_graph/components/splitters/llamaindex_splitter.py)
  - [Custom](./customize/build_graph/components/splitters/custom_splitter.py)
- [Chunk embedder]()
- Schema Builder:
  - [User-defined](./customize/build_graph/components/schema_builders/schema.py)
- Entity Relation Extractor:
  - [LLM-based](./customize/build_graph/components/extractors/llm_entity_relation_extractor.py)
  - [LLM-based with custom prompt](./customize/build_graph/components/extractors/llm_entity_relation_extractor_with_custom_prompt.py)
  - [Custom](./customize/build_graph/components/extractors/custom_extractor.py)
- Knowledge Graph Writer:
  - [Neo4j writer](./customize/build_graph/components/writers/neo4j_writer.py)
  - [Custom](./customize/build_graph/components/writers/custom_writer.py)
- Entity Resolver:
  - [SinglePropertyExactMatchResolver](./customize/build_graph/components/resolvers/simple_entity_resolver.py)
  - [SinglePropertyExactMatchResolver with pre-filter](./customize/build_graph/components/resolvers/simple_entity_resolver_pre_filter.py)
  - [Custom resolver](./customize/build_graph/components/resolvers/custom_resolver.py)
- [Custom component](./customize/build_graph/components/custom_component.py)


### Answer: GraphRAG

- [LangChain compatibility](./customize/answer/langchain_compatiblity.py)
- [Use a custom prompt](./customize/answer/custom_prompt.py)


## Database Operations

- [Create vector index](database_operations/create_vector_index.py)
- [Create full text index](create_fulltext_index.py)
- [Populate vector index](populate_vector_index.py)
