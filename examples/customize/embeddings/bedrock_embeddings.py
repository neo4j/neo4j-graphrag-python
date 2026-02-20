from botocore.exceptions import ClientError, NoCredentialsError, PartialCredentialsError

from neo4j_graphrag.embeddings import BedrockEmbeddings

# AWS credentials are read from environment or ~/.aws/credentials
embedder = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v2:0",
    dimensions=1024,
    region_name="us-east-1",
)

try:
    res = embedder.embed_query("my question")
    print(res[:10])
except NoCredentialsError:
    print("AWS credentials not found. Run 'aws configure' or set environment variables.")
except PartialCredentialsError as e:
    print(f"Incomplete AWS credentials: {e}")
except ClientError as e:
    print(f"AWS API error: {e}")
