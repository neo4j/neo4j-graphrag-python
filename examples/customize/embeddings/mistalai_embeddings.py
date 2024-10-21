"""This example demonstrate how to embed a text into a vector
using MistralAI models and API.
"""

from neo4j_graphrag.embeddings import MistralAIEmbeddings

# set api key here on in the MISTRAL_API_KEY env var
api_key = None
# api_key = "sk-..."

embeder = MistralAIEmbeddings(model="mistral-embed", api_key=api_key)
res = embeder.embed_query("my question")
print(res[:10])
