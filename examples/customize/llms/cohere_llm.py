from neo4j_graphrag.llm import CohereLLM, LLMResponse
from neo4j_graphrag.types import LLMMessage

# set api key here on in the CO_API_KEY env var
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

llm = CohereLLM(
    model_name="command-r",
    api_key=api_key,
)
res: LLMResponse = llm.invoke(input=messages)
print(res.content)
