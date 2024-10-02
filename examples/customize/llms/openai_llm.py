from neo4j_graphrag.llm import OpenAILLM, LLMResponse

# set api key here on in env var
api_key = None
# api_key = "sk-..."

llm = OpenAILLM(model_name="gpt-4o", api_key=api_key)
res: LLMResponse = llm.invoke("say something")
print(res.content)
