from neo4j_graphrag.llm import LLMResponse, OpenAILLM

# not used but needs to be provided
api_key = "ollama"

llm = OpenAILLM(
    base_url="http://localhost:11434/v1",
    model_name="<model_name>",
    api_key=api_key,
)
res: LLMResponse = llm.invoke("What is the additive color model?")
print(res.content)
