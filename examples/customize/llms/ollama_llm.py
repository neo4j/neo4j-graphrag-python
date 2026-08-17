"""This example demonstrate how to invoke an LLM using a local model
served by Ollama.

The model is a command-line argument, because which models you have depends on
what you have pulled locally. Pull one first, then name it:

    ollama pull llama3.2
    python examples/customize/llms/ollama_llm.py llama3.2
"""

import argparse

from neo4j_graphrag.llm import LLMResponse, OllamaLLM

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "model",
    nargs="?",
    default="llama3.2",
    help="name of a model you have pulled with `ollama pull` (default: %(default)s)",
)
args = parser.parse_args()

with OllamaLLM(
    model_name=args.model,
    # model_params={"options": {"temperature": 0}, "format": "json"},
    # host="...",  # if using a remote server
) as llm:
    res: LLMResponse = llm.invoke("What is the additive color model?")
    print(res.content)
