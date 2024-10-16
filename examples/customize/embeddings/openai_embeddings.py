"""This example demonstrate how to embed a text into a vector
using OpenAI models and API.
"""

from neo4j_graphrag.embeddings import OpenAIEmbeddings

# set api key here on in the OPENAI_API_KEY env var
api_key = None
# api_key = "sk-..."

embeder = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=api_key)
res = embeder.embed_query("my question")
print(res[:10])
