from neo4j_graphrag.llm import GeminiLLM

# set api key here on in the GOOGLE_API_KEY env var
api_key = None

llm = GeminiLLM(
    model_name="gemini-2.5-flash",
    api_key=api_key,
)
res = llm.invoke("say something")
print(res.content)
