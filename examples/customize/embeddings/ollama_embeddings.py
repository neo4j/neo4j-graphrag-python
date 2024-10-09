"""This example demonstrate how to embed a text into a vector
using OpenAI models and API.
"""

from neo4j_graphrag.embeddings import OpenAIEmbeddings

# not used but needs to be provided
api_key = "ollama"

embeder = OpenAIEmbeddings(
    base_url="http://localhost:11434/v1",
    api_key=api_key,
    model="<model_name>",
)
res = embeder.embed_query("my question")
print(res[:10])
