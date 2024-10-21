"""This example demonstrate how to embed a text into a vector
using Google models and the VertexAI API.
"""

from neo4j_graphrag.embeddings import VertexAIEmbeddings

embeder = VertexAIEmbeddings(model="text-embedding-004")
res = embeder.embed_query("my question")
print(res[:10])
