import os

from neo4j_graphrag.embeddings import GeminiEmbedder

api_key = os.getenv("GOOGLE_API_KEY")
assert api_key is not None, "you must set GOOGLE_API_KEY to run this experiment"

embedder = GeminiEmbedder(
    model="gemini-embedding-001",
    api_key=api_key,
)
res = embedder.embed_query("my question")
print(res[:10])
