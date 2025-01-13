from neo4j_graphrag.llm import AnthropicLLM, LLMResponse

# set api key here on in the ANTHROPIC_API_KEY env var
api_key = None

llm = AnthropicLLM(
    model_name="claude-3-opus-20240229",
    model_params={"max_tokens": 1000},  # max_tokens must be specified
    api_key=api_key,
)
res: LLMResponse = llm.invoke("say something")
print(res.content)
