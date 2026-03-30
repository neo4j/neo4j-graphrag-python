"""This example illustrates how to set system instructions for LLM.

OpenAILLM can be replaced by any supported LLM from this package.
"""

from neo4j_graphrag.llm import LLMResponse, OpenAILLM

# set api key here on in the OPENAI_API_KEY env var
api_key = None

question = "How fast is Santa Claus during the Christmas eve?"

with OpenAILLM(model_name="gpt-5", api_key=api_key) as llm:
    res: LLMResponse = llm.invoke(
        question,
        system_instruction="Answer with a serious tone",
    )
    print(res.content)
