from neo4j_graphrag.llm import LLMResponse, OpenAILLM
from neo4j_graphrag.message_history import InMemoryMessageHistory
from neo4j_graphrag.types import LLMMessage

# set api key here on in the OPENAI_API_KEY env var
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


llm = OpenAILLM(model_name="gpt-4o", api_key=api_key)
res: LLMResponse = llm.invoke(
    # "say something",
    # messages,
    InMemoryMessageHistory(
        messages=messages,
    )
)
print(res.content)
