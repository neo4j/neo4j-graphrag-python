import os

from neo4j_graphrag.llm import GeminiLLM

api_key = os.getenv("GOOGLE_API_KEY")
assert api_key is not None, "you must set GOOGLE_API_KEY to run this experiment"

llm = GeminiLLM(
    model_name="gemini-2.5-flash",
    api_key=api_key,
)
res = llm.invoke("say something")
print(res.content)
