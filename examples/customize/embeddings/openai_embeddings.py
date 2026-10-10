"""This example demonstrate how to embed a text into a vector
using OpenAI models and API.
"""

from neo4j_graphrag.embeddings import OpenAIEmbeddings

# set api key here or in the OPENAI_API_KEY env var
api_key = None
dimensions = 1536

embedder = OpenAIEmbeddings(
    model="text-embedding-3-large",
    dimensions=dimensions,
    api_key=api_key,
)
res = embedder.embed_query("my question")
print(res[:10])
