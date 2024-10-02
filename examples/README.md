# Examples Index

This folder contains examples usage for the different features
supported by the `neo4j-graphrag` package:

- Build Knowledge Graph from PDF or text
- Retrieve information from the graph
- Answer questions

Each of these steps have many customization options which
are listed in [the last section of this file](#customize).

## Build Knowledge Graph

- [End to end PDF to graph pipeline](build_graph/kg_builder_from_pdf.py)
- [End to end text to graph pipeline](build_graph/kg_builder_from_text.py)


## Retrieve

- [Retriever from an embedding vector](retrieve/similarity_search_for_vector.py)
- [Retriever from a text](retrieve/similarity_search_for_text.py)
- [Retriever with pre-filters](old/vector_search_with_filters.py)
- [Advanced retrieval with VectorCypherRetriever](retrieve/vector_cypher_retriever.py)
- [Hybrid retriever]()
- [Write a custom retriever]()


### External Retrievers

- [Weaviate](old/weaviate)
- [Pinecone](old/pinecone)
- [Qdrant]()


## Answer: GraphRAG

- [End to end GraphRAG](./answer/)


## Customize

### LLMs

- [OpenAI](./customize/llms/openai_llm.py)
- [Azure OpenAI]()
- [VertexAI]()
- [MistralAI]()
- [Ollama]()
- [Custom LLM](./customize/llms/custom_llm.py)


### Prompts

- [Using a custom prompt](old/graphrag_custom_prompt.py)


### Embedders

- [OpenAI]()
- [Azure OpenAI]()
- [VertexAI]()
- [MistalAI]()
- [Ollama]()
- [Custom LLM]()


### KG Construction

- [End to end example with explicit components]()

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
