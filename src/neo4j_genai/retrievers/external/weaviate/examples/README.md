### Start services locally

This is a manual task you need to do in the terminal.

This spins up Neo4j and Weaviate containers and is configuring Weaviate to use embeddings from Hugging Face's Sentence Transformers using the "all-MiniLM-L6-v2" model, which has 384 dimensions.

```bash
docker compose -f tests/e2e/docker-compose.yml up
```

### Write data (once)

Run this from the project root to write data to both dbs.

```
poetry run python tests/e2e/weaviate_e2e/populate_dbs.py
```

### Search

```
# search by vector
poetry run python src/neo4j_genai/retrievers/external/weaviate/examples/vector_search.py

# search by text, with embeddings generated locally (via embedder argument)
poetry run python src/neo4j_genai/retrievers/external/weaviate/examples/text_search_local_embedder.py

# search by text, with embeddings generated on the Weaviate side, via configured vectorizer
poetry run python src/neo4j_genai/retrievers/external/weaviate/examples/text_search_remote_embedder.py
```
