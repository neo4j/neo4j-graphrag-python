"""This example demonstrate how to embed a text into a vector
using a local model served by Ollama.

The model is a command-line argument, because which models you have depends on
what you have pulled locally. Pull an embedding model first, then name it:

    ollama pull nomic-embed-text
    python examples/customize/embeddings/ollama_embeddings.py nomic-embed-text
"""

import argparse

from neo4j_graphrag.embeddings import OllamaEmbeddings

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "model",
    nargs="?",
    default="nomic-embed-text",
    help="an embedding model you have pulled with `ollama pull` "
    "(default: %(default)s)",
)
args = parser.parse_args()

embeder = OllamaEmbeddings(
    model=args.model,
    # host="...",  # if using a remote server
)
res = embeder.embed_query("my question")
print(res[:10])
