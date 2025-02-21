import random
import string
from typing import Any, List, Optional, Union

from neo4j_graphrag.llm import LLMInterface, LLMResponse
from neo4j_graphrag.llm.types import LLMMessage
from neo4j_graphrag.message_history import MessageHistory


class CustomLLM(LLMInterface):
    def __init__(
        self, model_name: str, system_instruction: Optional[str] = None, **kwargs: Any
    ):
        super().__init__(model_name, **kwargs)

    def invoke(
        self,
        input: str,
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
    ) -> LLMResponse:
        content: str = (
            self.model_name + ": " + "".join(random.choices(string.ascii_letters, k=30))
        )
        return LLMResponse(content=content)

    async def ainvoke(
        self,
        input: str,
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
    ) -> LLMResponse:
        raise NotImplementedError()


llm = CustomLLM("")
res: LLMResponse = llm.invoke("text")
print(res.content)
