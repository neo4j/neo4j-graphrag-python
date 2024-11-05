"""This example demonstrate how to embed a text into a vector
using OpenAI models and API.
"""

from neo4j_graphrag.embeddings import AzureOpenAIEmbeddings

embedder = AzureOpenAIEmbeddings(
    model="text-embedding-ada-002",
    azure_endpoint="https://my-endpoint.openai.azure.com/",
    api_key="<my key>",
    api_version="<update version>",
)
res = embedder.embed_query("my question")
print(res[:10])
