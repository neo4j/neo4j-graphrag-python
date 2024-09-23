# Examples Index

## Knowledge Graph Creation

### End to end pipeline

- [End to end Pdf to graph pipeline](knowledge_graph_construction/kg_builder_from_pdf.py)
- [End to end text to graph pipeline](knowledge_graph_construction/kg_builder_from_text.py)

### Components

- Loaders:
  - [Load PDF file]()
  - [Custom]()
- Text Splitter:
  - [Fixed size splitter](./knowledge_graph_construction/components/splitters/fixed_size_splitter.py)
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


## LLMs

- [OpenAI](./llms/openai_llm.py)
- [Azure OpenAI]()
- [VertexAI]()
- [Ollama]()
- [Custom LLM](./llms/custom_llm.py)


## Embedders

- [OpenAI]()
- [Azure OpenAI]()
- [VertexAI]()
- [Ollama]()
- [Custom LLM]()


## GraphRAG

### End to end pipeline

- [End to end GraphRAG ](./graphrag/)
- [Retriever from an embedding vector](graphrag/retrievers/similarity_search_for_vector.py)
- [Retriever from a text](graphrag/retrievers/similarity_search_for_text.py)
- [Retriever with pre-filters](./vector_search_with_filters.py)

### Retrievers

- [Advanced retrieval with VectorCypherRetriever](./graphrag/retrievers/vector_cypher_retriever.py)
- [Hybrid retriever]()
- [Custom retriever]()

#### External Retrievers

- [Weaviate](./weaviate)
- [Pinecone](./pinecone)

### Prompts

- [Using a custom prompt](./graphrag_custom_prompt.py)


## Database Setup

- [Create vector index]()
- [Populate vector index]()
