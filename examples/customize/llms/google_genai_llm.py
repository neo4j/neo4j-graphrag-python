"""This example demonstrates how to invoke an LLM through the Gemini API.

Passing --base-url additionally demonstrates reaching a custom or self-hosted
Gemini-compatible endpoint instead of Google's default API:

    python examples/customize/llms/google_genai_llm.py
    python examples/customize/llms/google_genai_llm.py --base-url https://my-endpoint.example.com
"""

import argparse
import os

from neo4j_graphrag.llm import GeminiLLM

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--base-url",
    help="a Gemini-compatible endpoint to use instead of the default API",
)
args = parser.parse_args()

api_key = os.getenv("GOOGLE_API_KEY")
assert api_key is not None, "you must set GOOGLE_API_KEY to run this experiment"

llm = GeminiLLM(
    model_name="gemini-flash-latest",
    api_key=api_key,
)
res = llm.invoke("say something")
print(res.content)

# The genai SDK has no top-level base_url argument, so GeminiLLM applies it
# through `http_options` for you. Only exercised when you supply one, since
# there is no endpoint that would work for everyone.
if args.base_url:
    custom_llm = GeminiLLM(
        model_name="gemini-flash-latest",
        api_key=api_key,
        base_url=args.base_url,
    )
    res = custom_llm.invoke("say something")
    print(res.content)
