"""This example demonstrate how to invoke an LLM using a local model
served by Ollama.
"""

from neo4j_graphrag.llm import LLMResponse, OllamaLLM
from neo4j_graphrag.types import LLMMessage

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


llm = OllamaLLM(
    model_name="orca-mini:latest",
    # model_params={"options": {"temperature": 0}, "format": "json"},
    # host="...",  # if using a remote server
)
res: LLMResponse = llm.invoke(
    messages,
)
print(res.content)
