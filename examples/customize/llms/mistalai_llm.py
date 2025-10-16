from neo4j_graphrag.llm import MistralAILLM

# set api key here on in the MISTRAL_API_KEY env var
api_key = None

llm = MistralAILLM(
    model_name="mistral-small-latest",
    api_key=api_key,
)
llm.invoke("say something")
