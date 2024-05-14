### Start services locally

This is a manual task you need to do in the terminal.

```bash
docker run \
   --name testweaviate \
   --rm \
   -p8080:8080 -p 50051:50051 \
   cr.weaviate.io/semitechnologies/weaviate:1.25.1

docker run \
   --name testneo4j \
   --rm \
   -p7474:7474 -p7687:7687 \
   --env NEO4J_ACCEPT_LICENSE_AGREEMENT=eval \
   --env NEO4J_AUTH=neo4j/password \
   neo4j:enterprise
```

To run Weaviate with OpenAI Vectorizer enabled

```bash
docker run \
   --name testweaviate \
   --rm \
   -p8080:8080 -p 50051:50051 \
   --env ENABLE_MODULES=text2vec-openai \
   --env DEFAULT_VECTORIZER_MODULE=text2vec-openai \
   cr.weaviate.io/semitechnologies/weaviate:1.25.1
```

### Write data (once)

Run this from the project root to write data to both dbs

```
poetry run python tests/e2e/weaviate_e2e/populate_dbs.py
```

### Search

To run the text search examples you'd need to create a `.env` file and add a variable named `OPENAI_API_KEY=<your-api-key>` inside.

```
# search by vector
poetry run python src/neo4j_genai/retrievers/external/weaviate/examples/vector_search.py

# search by text, with embeddings generated locally (via embedder argument)
poetry run python src/neo4j_genai/retrievers/external/weaviate/examples/text_search_local_embedder.py

# search by text, with embeddings generated on the Weaviate side, via configured vectorizer
poetry run python src/neo4j_genai/retrievers/external/weaviate/examples/text_search_remote_embedder.py
```
