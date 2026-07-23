from neo4j_graphrag.llm import AnthropicLLM, LLMResponse

# set api key here on in the ANTHROPIC_API_KEY env var
api_key = None

with AnthropicLLM(
    model_name="claude-3-opus-20240229",
    model_params={"max_tokens": 1000},  # max_tokens must be specified
    api_key=api_key,
) as llm:
    res: LLMResponse = llm.invoke("say something")
    print(res.content)

# To reach a custom or self-hosted, Anthropic-compatible endpoint instead of
# Anthropic's default API, pass `base_url`. It's forwarded to both the sync
# and async SDK clients.
with AnthropicLLM(
    model_name="claude-3-opus-20240229",
    model_params={"max_tokens": 1000},
    api_key=api_key,
    base_url="https://my-custom-endpoint.example.com",
) as custom_llm:
    res = custom_llm.invoke("say something")
    print(res.content)
