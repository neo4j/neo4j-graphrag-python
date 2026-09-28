"""This example illustrates multi-turn conversations
by mocking a conversation between a user
and an LLM about Tom Hanks.

OpenAILLM can be replaced by any supported LLM from this package.
"""

from neo4j_graphrag.llm import LLMResponse, OpenAILLM

# set api key here on in the OPENAI_API_KEY env var
api_key = None

questions = [
    "What are some movies Tom Hanks starred in?",
    "Is he also a director?",
    "Wow, that's impressive. And what about his personal life, does he have children?",
]

history: list[dict[str, str]] = []
with OpenAILLM(model_name="gpt-5", api_key=api_key) as llm:
    for question in questions:
        messages = [*history, {"role": "user", "content": question}]
        res: LLMResponse = llm.invoke(messages)  # type: ignore[arg-type]
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": res.content})

        print("#" * 50, question)
        print(res.content)
        print("#" * 50)
