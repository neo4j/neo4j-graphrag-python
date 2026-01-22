### Usage Instructions

You will need both a Pinecone vector database and a Neo4j database to use this retriever.

### Writing Test Data

Update  `NEO4J_AUTH`, `NEO4J_URL`, and `PC_API_KEY` variables in the `tests/e2e/pinecone_e2e/populate_dbs.py` script then run this from the project root to write test data to both dbs.

```
uv run python -m tests/e2e/pinecone_e2e/populate_dbs.py
```

### Install Pinecone client

You need to install the `pinecone-client` package to use this retriever.

```bash
pip install pinecone-client
```

### Search
Update the `NEO4J_AUTH`, `NEO4J_URL`, and `PC_API_KEY` variables in each file then run one of the following from the project root to test the retriever.

```
# Search by vector
uv run python -m examples.customize.retrievers.external.pinecone.vector_search

# Search by text, with embeddings generated locally
uv run python -m examples.customize.retrievers.external.pinecone.text_search
```
