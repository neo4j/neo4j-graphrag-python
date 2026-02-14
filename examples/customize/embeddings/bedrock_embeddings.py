from neo4j_graphrag.embeddings import BedrockEmbeddings

# AWS credentials are read from environment or ~/.aws/credentials
embedder = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v2:0",
    dimensions=1024,
    region_name="us-east-1",
)
res = embedder.embed_query("my question")
print(res[:10])
