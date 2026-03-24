from neo4j_graphrag.embeddings import BedrockNovaEmbeddings

# Uses AWS credentials from environment variables, ~/.aws/credentials, or IAM role
embedder = BedrockNovaEmbeddings(
    model_id="amazon.nova-2-multimodal-embeddings-v1:0",  # default model
    region_name="us-east-1",
    embedding_dimension=1024,
    embedding_purpose="GENERIC_INDEX",
)
embedding = embedder.embed_query("Hello, world!")
print(f"Embedding dimension: {len(embedding)}")

# For retrieval queries, use a retrieval-optimized purpose
retrieval_embedder = BedrockNovaEmbeddings(
    region_name="us-east-1",
    embedding_purpose="TEXT_RETRIEVAL",
    embedding_dimension=1024,
)
query_embedding = retrieval_embedder.embed_query("What is GraphRAG?")
print(f"Query embedding dimension: {len(query_embedding)}")
