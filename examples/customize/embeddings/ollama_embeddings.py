"""This example demonstrate how to embed a text into a vector
using a local model served by Ollama.
"""

from neo4j_graphrag.embeddings import OllamaEmbeddings

embeder = OllamaEmbeddings(
    model="<model_name>",
    # host="...",  # if using a remote server
)
res = embeder.embed_query("my question")
print(res[:10])
