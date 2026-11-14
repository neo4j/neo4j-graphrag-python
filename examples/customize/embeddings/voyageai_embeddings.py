from neo4j_graphrag.embeddings import (
    VoyageAIContextualizedEmbeddings,
    VoyageAIEmbeddings,
)

# The API key is read from the VOYAGE_API_KEY env var (or pass `api_key=...`).

# The model is always explicit, there is no default.
embedder = VoyageAIEmbeddings(model="voyage-4", input_type="query")
res = embedder.embed_query("my question")
print(len(res), res[:10])

# Contextualized chunk embeddings: one string per document, chunked by the API.
contextualized = VoyageAIContextualizedEmbeddings(model="voyage-context-4")
documents = contextualized.embed_documents(["Neo4j is a graph database."])
print(documents[0].chunks, len(documents[0].embeddings[0]))
