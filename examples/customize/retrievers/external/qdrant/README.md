### Start services locally

Run the following command to spin up Neo4j and Qdrant containers.

```bash
docker compose -f tests/e2e/docker-compose.yml up
```

### Write data (once)

Run this from the project root to write data to both Neo4J and Qdrant.

```bash
poetry run python tests/e2e/qdrant_e2e/populate_dbs.py
```

### Install Qdrant client

```bash
pip install qdrant-client
```

### Search

```bash
# search by vector
poetry run python -m examples.qdrant.vector_search

# search by text, with embeddings generated locally
poetry run python -m examples.qdrant.text_search
```
