from neo4j_graphrag.llm import CohereLLM, LLMResponse

# set api key here on in the CO_API_KEY env var
api_key = None

llm = CohereLLM(
    model_name="command-r",
    api_key=api_key,
)
res: LLMResponse = llm.invoke("say something")
print(res.content)
