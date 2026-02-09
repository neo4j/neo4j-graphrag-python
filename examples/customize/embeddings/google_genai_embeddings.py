from neo4j_graphrag.embeddings import GeminiEmbedder

# set api key here on in the GOOGLE_API_KEY env var
api_key = None

embedder = GeminiEmbedder(
    model="gemini-embedding-001",
    api_key=api_key,
)
res = embedder.embed_query("my question")
print(res[:10])
