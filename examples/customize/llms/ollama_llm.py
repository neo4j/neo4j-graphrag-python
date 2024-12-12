"""This example demonstrate how to invoke an LLM using a local model
served by Ollama.
"""

from neo4j_graphrag.llm import LLMResponse, OllamaLLM

llm = OllamaLLM(
    model_name="<model_name>",
)
res: LLMResponse = llm.invoke("What is the additive color model?")
print(res.content)
