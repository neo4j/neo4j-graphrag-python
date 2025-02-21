"""This example illustrates the message_history feature
of the LLMInterface by mocking a conversation between a user
and an LLM about Tom Hanks.

Neo4j is used as the database for storing the message history.

OpenAILLM can be replaced by any supported LLM from this package.
"""

import neo4j
from neo4j_graphrag.llm import LLMResponse, OpenAILLM
from neo4j_graphrag.message_history import Neo4jMessageHistory

# Define database credentials
URI = "neo4j+s://demo.neo4jlabs.com"
AUTH = ("recommendations", "recommendations")
DATABASE = "recommendations"
INDEX = "moviePlotsEmbedding"

# set api key here on in the OPENAI_API_KEY env var
api_key = None

llm = OpenAILLM(model_name="gpt-4o", api_key=api_key)

questions = [
    "What are some movies Tom Hanks starred in?",
    "Is he also a director?",
    "Wow, that's impressive. And what about his personal life, does he have children?",
]

driver = neo4j.GraphDatabase.driver(
    URI,
    auth=AUTH,
    database=DATABASE,
)

history = Neo4jMessageHistory(session_id="123", driver=driver, window=10)

for question in questions:
    res: LLMResponse = llm.invoke(
        question,
        message_history=history,
    )
    history.add_message(
        {
            "role": "user",
            "content": question,
        }
    )
    history.add_message(
        {
            "role": "assistant",
            "content": res.content,
        }
    )

    print("#" * 50, question)
    print(res.content)
    print("#" * 50)
