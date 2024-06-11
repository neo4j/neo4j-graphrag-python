# @neo4j/neo4j-genai-python

## Next

## 0.2.0

### Fixed

-   Removed Pinecone and Weaviate retrievers from **init**.py to prevent ImportError when optional dependencies are not installed.

## 0.2.0a5

## 0.2.0a3

### Added

-   Introduced custom exceptions for improved error handling, including `RetrieverInitializationError`, `SearchValidationError`, `FilterValidationError`, `EmbeddingRequiredError`, `RecordCreationError`, `Neo4jIndexError`, and `Neo4jVersionError`.
-   Retrievers that integrates with a Weaviate vector database: `WeaviateNeo4jRetriever`.
-   New return types that help with getting retriever results: `RetrieverResult` and `RetrieverResultItem`.
-   Supported wrapper embedder object for sentence-transformers embeddings: `SentenceTransformerEmbeddings`.
-   `Text2CypherRetriever` object which allows for the retrieval of records from a Neo4j database using natural language.

### Changed

-   Replaced `ValueError` with custom exceptions across various modules for clearer and more specific error messages.

### Fixed

-   Updated documentation to include new custom exceptions.
-   Improved the use of Pydantic for input data validation for retriever objects.
