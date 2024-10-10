# Examples Index

This folder contains examples usage for the different features
supported by the `neo4j-graphrag` package:

- Build Knowledge Graph from PDF or text
- Retrieve information from the graph
- Answer questions

Each of these steps have many customization options which
are listed in [the last section of this file](#customize).

## Build Knowledge Graph

- [End to end PDF to graph simple pipeline](build_graph/simple_kg_builder_from_pdf.py)
- [End to end text to graph simple pipeline](build_graph/simple_kg_builder_from_text.py)


## Retrieve

- [Retriever from an embedding vector](retrieve/similarity_search_for_vector.py)
- [Retriever from a text](retrieve/similarity_search_for_text.py)
- [Retriever with pre-filters](customize/retrievers/use_pre_filters.py)
- [Advanced retrieval with VectorCypherRetriever](retrieve/vector_cypher_retriever.py)
- [Hybrid retriever]()
- [Write a custom retriever]()


### External Retrievers

#### Weaviate

- [Vector search](customize/retrievers/external/weaviate/vector_search.py)
- [Text search with local embeder](customize/retrievers/external/weaviate/text_search_local_embedder.py)
- [Text search with remote embeder](customize/retrievers/external/weaviate/text_search_remote_embedder.py)

#### Pinecone

- [Vector search](./customize/retrievers/external/pinecone/vector_search.py)
- [Text search](./customize/retrievers/external/pinecone/text_search.py)


### Qdrant

- [Qdrant]()


## Answer: GraphRAG

- [End to end GraphRAG](./answer/graphrag.py)


## Customize

### Retriever

- [Control result format](customize/retrievers/result_formatter.py) (for `VectorRetriever`, `HybridRetriever`, `VectorCypherRetriever` and `HybridCypherRetriever`)

### LLMs

- [OpenAI (GPT)](./customize/llms/openai_llm.py)
- [Azure OpenAI]()
- [VertexAI (Gemini)](./customize/llms/vertexai_llm.py)
- [MistralAI](./customize/llms/mistalai_llm.py)
- [Cohere](./customize/llms/cohere_llm.py)
- [Anthropic (Claude)](./customize/llms/anthropic_llm.py)
- [Ollama]()
- [Custom LLM](./customize/llms/custom_llm.py)


### Prompts

- [Using a custom prompt](old/graphrag_custom_prompt.py)


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


## Database Setup

- [Create vector index]()
- [Create full text index]()
- [Populate vector index]()
