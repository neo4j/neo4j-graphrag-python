"""This example demonstrate how to embed a text into a vector
using MistralAI models and API.
"""

from neo4j_graphrag.embeddings import MistralAIEmbeddings

# set api key here on in the MISTRAL_API_KEY env var
api_key = None

embeder = MistralAIEmbeddings(model="mistral-embed", api_key=api_key)
res = embeder.embed_query(
    "my question",
    # optionally, set output dimensions
    dimensions=256,
)
print("Embedding dimensions", len(res))
print(res[:10])
