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

# To reach a custom or self-hosted, Gemini-compatible endpoint instead of
# Google's default API, pass `base_url`. The genai SDK has no top-level
# base_url argument, so GeminiLLM applies it through `http_options` for you.
custom_llm = GeminiLLM(
    model_name="gemini-2.5-flash",
    api_key=api_key,
    base_url="https://my-custom-endpoint.example.com",
)
res = custom_llm.invoke("say something")
print(res.content)
