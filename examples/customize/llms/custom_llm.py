import random
import string
from typing import Any

from neo4j_graphrag.llm import LLMInterface, LLMResponse


class CustomLLM(LLMInterface):
    def __init__(self, model_name: str, **kwargs: Any):
        super().__init__(model_name, **kwargs)

    def invoke(self, input: str) -> LLMResponse:
        content: str = (
            self.model_name + ": " + "".join(random.choices(string.ascii_letters, k=30))
        )
        return LLMResponse(content=content)

    async def ainvoke(self, input: str) -> LLMResponse:
        raise NotImplementedError()


llm = CustomLLM("")
res: LLMResponse = llm.invoke("text")
print(res.content)
