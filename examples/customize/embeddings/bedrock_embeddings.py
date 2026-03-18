from neo4j_graphrag.embeddings import BedrockEmbeddings

# Uses AWS credentials from environment variables, ~/.aws/credentials, or IAM role
embedder = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v2:0",  # default model
    region_name="us-east-1",
)
embedding = embedder.embed_query("Hello, world!")
print(f"Embedding dimension: {len(embedding)}")
