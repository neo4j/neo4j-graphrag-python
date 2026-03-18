# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the **official Neo4j GraphRAG package for Python** - a library for building Graph Retrieval Augmented Generation (GraphRAG) applications. It enables:
- Building knowledge graphs from text/PDFs using LLMs
- Retrieving information from Neo4j graphs with various retriever strategies
- Generating answers using RAG patterns with LLMs

**Python:** 3.10-3.14 (spaCy features require 3.13 or earlier)

## Common Commands

```bash
# Install dependencies
uv sync --group dev              # Development setup
uv sync --all-extras             # All optional dependencies

# Testing
uv run pytest tests/unit                    # Unit tests
uv run pytest tests/unit -k "test_name"     # Single test
uv run pytest tests/e2e                     # E2E tests (requires Docker)

# Start E2E services
docker compose -f tests/e2e/docker-compose.yml up

# Code quality
uv run ruff format .             # Format code
uv run ruff check .              # Lint
uv run mypy .                    # Type check
pre-commit run --all-files       # All checks

# Pre-commit setup
pre-commit install
```

## Architecture

### Core Package Structure (`src/neo4j_graphrag/`)

**Retrievers** (`retrievers/`): Chain-of-responsibility pattern for different retrieval strategies
- `VectorRetriever` - Vector similarity search
- `HybridRetriever` - Combined vector + fulltext search
- `Text2CypherRetriever` - LLM generates Cypher queries
- `external/` - Weaviate, Pinecone, Qdrant integrations

**LLM/Embeddings** (`llm/`, `embeddings/`): Plugin pattern for providers
- All providers implement `LLMInterface` or `EmbedderInterface`
- Supported: OpenAI, VertexAI, Anthropic, Cohere, Mistral, Ollama, sentence-transformers

**Generation** (`generation/`): RAG orchestration
- `GraphRAG` class coordinates retriever → LLM pipeline
- Template-based prompt management

**Experimental** (`experimental/`): Knowledge graph construction pipelines
- `SimpleKGPipeline` - Simplified KG building from text/PDFs
- `Pipeline` - DAG-based component orchestration (loaders → splitters → extractors → writers)
- Requires APOC core library in Neo4j

### Key Design Patterns

1. **Pydantic validation throughout** - All configurations and data models use Pydantic v2
2. **Async-first** - Methods provide `run_async()` variants
3. **Neo4j driver integration** - Proper lifecycle management, version checking (requires 5.18.1+)
4. **Rate limiting** - Configurable handlers with tenacity for API calls

### Important Files

- `schema.py` - Core Pydantic models for validation
- `types.py` - TypedDict/Enum definitions (LLMMessage, EntityType, SearchType)
- `retrievers/base.py` - Abstract Retriever base class
- `llm/base.py` - Abstract LLM interface

## Testing Notes

- Unit tests have no external dependencies
- E2E tests require Docker with Neo4j and Weaviate running
- Cross-version testing via tox (py310-py314)

## Optional Dependencies

Install specific providers as needed:
```bash
pip install "neo4j-graphrag[openai]"           # OpenAI
pip install "neo4j-graphrag[anthropic]"        # Anthropic
pip install "neo4j-graphrag[experimental]"     # KG builder pipelines
pip install "neo4j-graphrag[weaviate,pinecone]" # External vector DBs
```
