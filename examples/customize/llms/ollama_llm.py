"""This example demonstrate how to invoke an LLM using a local model
served by Ollama.
"""

from neo4j_graphrag.llm import LLMResponse, OllamaLLM

with OllamaLLM(
    model_name="<model_name>",
    # model_params={"options": {"temperature": 0}, "format": "json"},
    # host="...",  # if using a remote server
) as llm:
    res: LLMResponse = llm.invoke("What is the additive color model?")
    print(res.content)
