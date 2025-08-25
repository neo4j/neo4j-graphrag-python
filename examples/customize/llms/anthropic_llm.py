from neo4j_graphrag.llm import AnthropicLLM, LLMResponse
from neo4j_graphrag.types import LLMMessage

# set api key here on in the ANTHROPIC_API_KEY env var
api_key = None

messages: list[LLMMessage] = [
    {
        "role": "system",
        "content": "You are a seasoned actor and expert performer, renowned for your one-man shows and comedic talent.",
    },
    {
        "role": "user",
        "content": "say something",
    },
]


llm = AnthropicLLM(
    model_name="claude-3-opus-20240229",
    model_params={"max_tokens": 1000},  # max_tokens must be specified
    api_key=api_key,
)
res: LLMResponse = llm.invoke(
    # "say something",
    messages,
)
print(res.content)
