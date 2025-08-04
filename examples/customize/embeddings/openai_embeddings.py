"""This example demonstrate how to embed a text into a vector
using OpenAI models and API.
"""

from neo4j_graphrag.embeddings import OpenAIEmbeddings

# set api key here on in the OPENAI_API_KEY env var
api_key = None

embeder = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
res = embeder.embed_query(
    "my question",
    # optionally, set output dimensions
    # dimensions=256,
)

print("Embedding dimensions", len(res))
print(res[:10])
