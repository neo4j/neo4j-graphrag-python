# @neo4j/neo4j-graphrag-python

## Next

### Added
- Made `relations` and `potential_schema` optional in `SchemaBuilder`.
- Added a check to prevent the use of deprecated Cypher syntax for Neo4j versions 5.23.0 and above.

### Changed
- Vector and Hybrid retrievers used with `return_properties` now also return the node labels (`nodeLabels`) and the node's element ID (`id`).
- `HybridRetriever` now filters out the embedding property index in `self.vector_index_name` from the retriever result by default.


## 1.1.0

### Added
- Introduced a `fail_if_exist` option to index creation functions to control behavior when an index already exists.
- Added Qdrant retriever in neo4j_graphrag.retrievers.

### Changed
- Comprehensive rewrite of the README to improve clarity and provide detailed usage examples.

## 1.0.0

### Fixed
- Fix a bug where `openai` Python client and `numpy` were required to import any embedder or LLM.

### Changed
- The value associated to the enum field `OnError.IGNORE` has been changed from "CONTINUE" to "IGNORE" to stick to the convention and match the field name.

### Added
- Added `SinglePropertyExactMatchResolver` component allowing to merge entities with exact same property (e.g. name)
- Added the `SimpleKGPipeline` class, a simplified abstraction layer to streamline knowledge graph building processes from text documents.

## 1.0.0a1

## 1.0.0a0

### Added
- Added `SinglePropertyExactMatchResolver` component allowing to merge entities with exact same property (e.g. name)

## 0.7.0

### Added
- Added AzureOpenAILLM and AzureOpenAIEmbeddings to support Azure served OpenAI models
- Added `template` validation in `PromptTemplate` class upon construction.
- Examples demonstrating the use of Mistral embeddings and LLM in RAG pipelines.
- Added feature to include kwargs in `Text2CypherRetriever.search()` that will be injected into a custom prompt, if provided.
- Added validation to `custom_prompt` parameter of `Text2CypherRetriever` to ensure that `query_text` placeholder exists in prompt.
- Introduced a fixed size text splitter component for splitting text into specified fixed size chunks with overlap. Updated examples and tests to utilize this new component.
- Introduced Vertex AI LLM class for integrating Vertex AI models.
- Added unit tests for the Vertex AI LLM class.
- Added support for Cohere LLM and embeddings - added optional dependency to `cohere`.
- Added support for Anthropic LLM - added optional dependency to `anthropic`.
- Added support for MistralAI LLM - added optional dependency to `mistralai`.
- Added support for Qdrant - added optional dependency to `qdrant-client`.

### Fixed
- Resolved import issue with the Vertex AI Embeddings class.
- Fixed bug in `Text2CypherRetriever` using `custom_prompt` arg where the `search` method would not inject the `query_text` content.
- `custom_prompt` arg is now converted to `Text2CypherTemplate` class within the `Text2CypherRetriever.get_search_results` method.
- `Text2CypherTemplate` and `RAGTemplate` prompt templates now require `query_text` arg and will error if it is not present. Previous `query_text` aliases may be used, but will warn of deprecation.
- Resolved issue where Neo4jWriter component would raise an error if the start or end node ID was not defined properly in the input.
- Resolved issue where relationship types was not escaped in the insert Cypher query.
- Improved query performance in Neo4jWriter: created nodes now have a generic `__KGBuilder__` label and an index is created on the `__KGBuilder__.id` property. Moreover, insertion queries are now batched. Batch size can be controlled using the `batch_size` parameter in the `Neo4jWriter` component.

### Changed
- Moved the Embedder class to the neo4j_graphrag.embeddings directory for better organization alongside other custom embedders.
- Removed query argument from the GraphRAG class' `.search` method; users must now use `query_text`.
- Neo4jWriter component now runs a single query to merge node and set its embeddings if any.
- Nodes created by the `Neo4jWriter` now have an extra `__KGBuilder__` label. Nodes from the entity graph also have an `__Entity__` label.
- Dropped support for Python 3.8 (end of life).

## 0.6.3
### Changed
- Updated documentation links in README.
- Renamed deprecated package references in documentation.

### Added
- Introduction page to the documentation content tree.
- Introduced a new Vertex AI embeddings class for generating text embeddings using Vertex AI.
- Updated documentation to include OpenAI and Vertex AI embeddings classes.
- Added google-cloud-aiplatform as an optional dependency for Vertex AI embeddings.

### Fixed
- Make `pygraphviz` an optional dependency - it is now only required when calling `pipeline.draw`.

## 0.6.2

### Fixed
- Moved pygraphviz to optional dependencies under [tool.poetry.extras] in pyproject.toml to resolve an issue where pip install neo4j-graphrag incorrectly required pygraphviz as a mandatory dependency.

## 0.6.1

### Changed
- Officially renamed neo4j-genai to neo4j-graphrag. For the final release version of neo4j-genai, please visit https://pypi.org/project/neo4j-genai/.

## 0.6.0

### IMPORTANT NOTICE
- The `neo4j-genai` package is now deprecated. Users are advised to switch to the new package `neo4j-graphrag`.
### Added
- Ability to visualise pipeline with `my_pipeline.draw("pipeline.png")`

### Fixed
- Pipelines now return correct results when the same pipeline is run in parallel.

### Changed
- Pipeline run method now return a PipelineResult object.
- Improved parameter validation for pipelines (#124). Pipeline now raise an error before a run starts if:
  - the same parameter is mapped twice
  - or a parameter is defined in the mapping but is not a valid component input


## 0.5.0

### Added
- PDF-to-graph pipeline for knowledge graph construction in experimental mode
- Introduced support for Component/Pipeline flexible architecture.
- Added new components for knowledge graph construction, including text splitters, schema builders, entity-relation extractors, and Neo4j writers.
- Implemented end-to-end tests for the new knowledge graph builder pipeline.

### Changed
- When saving the lexical graph in a KG creation pipeline, the document is also saved as a specific node, together with relationships between each chunk and the document they were created from.

### Fixed
- Corrected the hybrid retriever query to ensure proper normalization of scores in vector search results.

## 0.4.0

### Added
- Add optional custom_prompt arg to the Text2CypherRetriever class.

### Changed
- `GraphRAG.search` method first parameter has been renamed `query_text` (was `query`) for consistency with the retrievers interface.
- Made `GraphRAG.search` method backwards compatible with the query parameter, raising warnings to encourage using query_text instead.

## 0.3.1

### Fixed
-   Corrected initialization to allow specifying the embedding model name.
-   Removed sentence_transformers from embeddings/__init__.py to avoid ImportError when the package is not installed.

## 0.3.0

### Added
-   Stopped embeddings from being returned when searching with `VectorRetriever`. Added `nodeLabels` and `id` to the metadata of `VectorRetriever` results.
-   Added `upsert_vector` utility function for attaching vectors to node properties.
-   Introduced `Neo4jInsertionError` for handling insertion failures in Neo4j.
-   Included Pinecone and Weaviate retrievers in neo4j_graphrag.retrievers.
-   Introduced the GraphRAG object, enabling a full RAG (Retrieval-Augmented Generation) pipeline with context retrieval, prompt formatting, and answer generation.
-   Added PromptTemplate and RagTemplate for customizable prompt generation.
-   Added LLMInterface with implementation for OpenAI LLM.
-   Updated project configuration to support multiple Python versions (3.8 to 3.12) in CI workflows.
-   Improved developer experience by copying the docstring from the `Retriever.get_search_results` method to the `Retriever.search` method
-   Support for specifying database names in index handling methods and retrievers.
-   User Guide in documentation.
-   Introduced result_formatter argument to all retrievers, allowing custom formatting of retriever results.

### Changed
-   Refactored import paths for retrievers to neo4j_graphrag.retrievers.
-   Implemented exception chaining for all re-raised exceptions to improve stack trace readability.
-   Made error messages in `index.py` more consistent.
-   Renamed `Retriever._get_search_results` to `Retriever.get_search_results`
-   Updated retrievers and index handling methods to accept optional database names.

## 0.2.0

### Fixed

-   Removed Pinecone and Weaviate retrievers from **init**.py to prevent ImportError when optional dependencies are not installed.
-   Moved few-shot examples in `Text2CypherRetriever` to the constructor for better initialization and usage. Updated unit tests and example script accordingly.
-   Fixed regex warnings in E2E tests for Weaviate and Pinecone retrievers.
-   Corrected HuggingFaceEmbeddings import in E2E tests.


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
