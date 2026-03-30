from neo4j_graphrag.llm import LLMResponse, OpenAILLM

# set api key here on in the OPENAI_API_KEY env var
api_key = None

with OpenAILLM(model_name="gpt-5", api_key=api_key) as llm:
    res: LLMResponse = llm.invoke("say something")
    print(res.content)
