### Start services locally

Run the following command to spin up Neo4j and Qdrant containers. Qdrant is behind the
`vectordb` profile, so it needs the flag.

```bash
docker compose -f tests/e2e/docker-compose.yml --profile vectordb up -d --wait
```

### Write data (once)

Run this from the project root to write data to both Neo4J and Qdrant.

```bash
uv run python -m examples.customize.retrievers.external.qdrant.populate_dbs
```

### Install Qdrant client

```bash
pip install qdrant-client
```

### Search

```bash
# search by vector
uv run python -m examples.customize.retrievers.external.qdrant.vector_search

# search by text, with embeddings generated locally
uv run python -m examples.customize.retrievers.external.qdrant.text_search
```
