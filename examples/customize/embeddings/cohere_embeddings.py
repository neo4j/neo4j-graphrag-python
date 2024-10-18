from neo4j_graphrag.embeddings import CohereEmbeddings

# set api key here on in the CO_API_KEY env var
api_key = None
# api_key = "sk-..."

embeder = CohereEmbeddings(
    model="embed-english-v3.0",
    api_key=api_key,
)
res = embeder.embed_query("my question")
print(res[:10])
