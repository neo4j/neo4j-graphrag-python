import random
import string
from typing import Any, Awaitable, Callable, List, Optional, TypeVar, Union

from neo4j_graphrag.llm import LLMInterface, LLMResponse
from neo4j_graphrag.llm.rate_limit import (
    RateLimitHandler,
    # rate_limit_handler,
    # async_rate_limit_handler,
)
from neo4j_graphrag.message_history import MessageHistory
from neo4j_graphrag.types import LLMMessage


class CustomLLM(LLMInterface):
    def __init__(
        self, model_name: str, system_instruction: Optional[str] = None, **kwargs: Any
    ):
        super().__init__(model_name, **kwargs)

    # Optional: Apply rate limit handling to synchronous invoke method
    # @rate_limit_handler
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

    # Optional: Apply rate limit handling to asynchronous ainvoke method
    # @async_rate_limit_handler
    async def ainvoke(
        self,
        input: str,
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
    ) -> LLMResponse:
        raise NotImplementedError()


llm = CustomLLM(
    ""
)  # if rate_limit_handler and async_rate_limit_handler decorators are used, the default rate limit handler will be applied automatically (retry with exponential backoff)
res: LLMResponse = llm.invoke("text")
print(res.content)

# If rate_limit_handler and async_rate_limit_handler decorators are used and you want to use a custom rate limit handler
# Type variables for function signatures used in rate limit handlers
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
