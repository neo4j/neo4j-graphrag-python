import random
import string
from typing import Any, Awaitable, Callable, Optional, TypeVar

from neo4j_graphrag.llm import LLMInterface, LLMResponse
from neo4j_graphrag.utils.rate_limit import (
    RateLimitHandler,
    # rate_limit_handler,
    # async_rate_limit_handler,
)
from neo4j_graphrag.types import LLMMessage


class CustomLLM(LLMInterface):
    def __init__(
        self, model_name: str, system_instruction: Optional[str] = None, **kwargs: Any
    ):
        super().__init__(model_name, **kwargs)

    def _invoke(
        self,
        input: list[LLMMessage],
    ) -> LLMResponse:
        content: str = (
            self.model_name + ": " + "".join(random.choices(string.ascii_letters, k=30))
        )
        return LLMResponse(content=content)

    async def _ainvoke(
        self,
        input: list[LLMMessage],
    ) -> LLMResponse:
        raise NotImplementedError()


llm = CustomLLM("")
res: LLMResponse = llm.invoke("text")
print(res.content)

F = TypeVar("F", bound=Callable[..., Any])
AF = TypeVar("AF", bound=Callable[..., Awaitable[Any]])


class CustomRateLimitHandler(RateLimitHandler):
    def __init__(self) -> None:
        super().__init__()

    def handle_sync(self, func: F) -> F:
        # error handling here
        return func

    def handle_async(self, func: AF) -> AF:
        # error handling here
        return func


llm_with_custom_rate_limit_handler = CustomLLM(
    "", rate_limit_handler=CustomRateLimitHandler()
)
result: LLMResponse = llm_with_custom_rate_limit_handler.invoke("text")
print(result.content)
