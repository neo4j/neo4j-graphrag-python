"""This example illustrates the message_history feature
of the LLMInterface by mocking a conversation between a user
and an LLM about Tom Hanks.

OpenAILLM can be replaced by any supported LLM from this package.
"""

from neo4j_graphrag.llm import LLMResponse, OpenAILLM

# set api key here on in the OPENAI_API_KEY env var
api_key = None

llm = OpenAILLM(model_name="gpt-4o", api_key=api_key)

questions = [
    "What are some movies Tom Hanks starred in?",
    "Is he also a director?",
    "Wow, that's impressive. And what about his personal life, does he have children?",
]

history: list[dict[str, str]] = []
for question in questions:
    res: LLMResponse = llm.invoke(
        question,
        message_history=history,  # type: ignore
    )
    history.append(
        {
            "role": "user",
            "content": question,
        }
    )
    history.append(
        {
            "role": "assistant",
            "content": res.content,
        }
    )

    print("#" * 50, question)
    print(res.content)
    print("#" * 50)
