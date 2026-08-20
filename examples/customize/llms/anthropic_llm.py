"""This example demonstrates how to invoke an LLM through Anthropic.

Passing --base-url additionally demonstrates reaching a custom or self-hosted
Anthropic-compatible endpoint instead of Anthropic's default API:

    python examples/customize/llms/anthropic_llm.py
    python examples/customize/llms/anthropic_llm.py --base-url https://my-endpoint.example.com
"""

import argparse

from neo4j_graphrag.llm import AnthropicLLM, LLMResponse

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--base-url",
    help="an Anthropic-compatible endpoint to use instead of the default API",
)
args = parser.parse_args()

# set api key here on in the ANTHROPIC_API_KEY env var
api_key = None

with AnthropicLLM(
    model_name="claude-sonnet-4-5",
    model_params={"max_tokens": 1000},  # max_tokens must be specified
    api_key=api_key,
) as llm:
    res: LLMResponse = llm.invoke("say something")
    print(res.content)

# `base_url` is forwarded to both the sync and async SDK clients. It is only
# exercised when you supply one, since there is no endpoint that would work for
# everyone.
if args.base_url:
    with AnthropicLLM(
        model_name="claude-sonnet-4-5",
        model_params={"max_tokens": 1000},
        api_key=api_key,
        base_url=args.base_url,
    ) as custom_llm:
        res = custom_llm.invoke("say something")
        print(res.content)
