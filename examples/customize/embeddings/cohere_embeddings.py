from neo4j_graphrag.embeddings import CohereEmbeddings

# set api key here on in the CO_API_KEY env var
api_key = None

embeder = CohereEmbeddings(
    model="embed-v4.0",
    api_key=api_key,
)
res = embeder.embed_query(
    "my question",
    # optionally, set output dimensions if it's supported by the model
    dimensions=256,
    input_type="search_query",
)
print("Embedding dimensions", len(res))
print(res[:10])
