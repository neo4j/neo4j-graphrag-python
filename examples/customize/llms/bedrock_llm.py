from neo4j_graphrag.llm import BedrockLLM

# AWS credentials are read from environment or ~/.aws/credentials
llm = BedrockLLM(
    model_name="us.anthropic.claude-sonnet-4-20250514-v1:0",
    model_params={"temperature": 0.7, "maxTokens": 1024},
    region_name="us-east-1",
)
res = llm.invoke("say something")
print(res.content)
