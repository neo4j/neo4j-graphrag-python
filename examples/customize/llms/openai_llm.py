from neo4j_graphrag.llm import LLMResponse, OpenAILLM

# set api key here on in the OPENAI_API_KEY env var
api_key = None

llm = OpenAILLM(model_name="gpt-4o", api_key=api_key)
res: LLMResponse = llm.invoke("say something")
print(res.content)
