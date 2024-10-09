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
- [Retriever with pre-filters](old/vector_search_with_filters.py)
- [Advanced retrieval with VectorCypherRetriever](retrieve/vector_cypher_retriever.py)
- [Hybrid retriever]()
- [Write a custom retriever]()


### External Retrievers

#### Weaviate

- [Vector search](customize/retrievers/external/weaviate/vector_search.py)
- [Text search with local embeder](customize/retrievers/external/weaviate/text_search_local_embedder.py)
- [Text search with remote embeder](customize/retrievers/external/weaviate/text_search_remote_embedder.py)

#### Pinecone

- [Pinecone](old/pinecone)

### Qdrant

- [Qdrant]()


## Answer: GraphRAG

- [End to end GraphRAG](./answer/)


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

### Components

- Loaders:
  - [Load PDF file]()
  - [Custom]()
- Text Splitter:
  - [Fixed size splitter](./customize/build_graph/components/splitters/fixed_size_splitter.py)
  - [Splitter from LangChain]()
  - [Splitter from LLamaIndex]()
  - [Custom]()
- [Chunk embedder]()
- Schema Builder:
  - [User-defined]()
  - [Custom]()
- Entity Relation Extractor:
  - [LLM-based]()
  - [Custom]()
- Knowledge Graph Writer:
  - [Neo4j writer]()
  - [Custom]()
- Entity Resolver:
  - [...]()


## Database Setup

- [Create vector index]()
- [Create full text index]()
- [Populate vector index]()
