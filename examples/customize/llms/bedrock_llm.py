from neo4j_graphrag.llm import BedrockLLM, LLMResponse

# Uses AWS credentials from environment variables, ~/.aws/credentials, or IAM role
# model_id uses inference profile format: {region}.{provider}.{model}
# See: https://docs.aws.amazon.com/bedrock/latest/userguide/inference-profiles-support.html
llm = BedrockLLM(
    model_id="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    region_name="us-east-1",
    # model_params={"temperature": 0.7, "maxTokens": 1000},
)
res: LLMResponse = llm.invoke("say something")
print(res.content)
