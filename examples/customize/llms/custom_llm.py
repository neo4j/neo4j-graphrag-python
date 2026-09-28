import random
import string
from typing import Any, Awaitable, Callable, List, Optional, Type, TypeVar, Union

from pydantic import BaseModel

from neo4j_graphrag.llm import LLMBase, LLMResponse
from neo4j_graphrag.utils.rate_limit import (
    RateLimitHandler,
    # rate_limit_handler,
    # async_rate_limit_handler,
)
from neo4j_graphrag.types import LLMMessage
from neo4j_graphrag.exceptions import RetryableError


class CustomLLM(LLMBase):
    def __init__(
        self, model_name: str, system_instruction: Optional[str] = None, **kwargs: Any
    ):
        super().__init__(model_name, **kwargs)

    # Optional: Apply rate limit handling to synchronous invoke method
    # @rate_limit_handler
    def invoke(
        self,
        input: List[LLMMessage],
        *,
        response_format: Optional[Union[Type[BaseModel], dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        content: str = (
            self.model_name + ": " + "".join(random.choices(string.ascii_letters, k=30))
        )
        return LLMResponse(content=content)

    # Optional: Apply rate limit handling to asynchronous ainvoke method
    # @async_rate_limit_handler
    async def ainvoke(
        self,
        input: List[LLMMessage],
        *,
        response_format: Optional[Union[Type[BaseModel], dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        raise NotImplementedError()


llm = CustomLLM(
    ""
)  # if rate_limit_handler and async_rate_limit_handler decorators are used, the default rate limit handler will be applied automatically (retry with exponential backoff)
res: LLMResponse = llm.invoke([{"role": "user", "content": "text"}])
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

    def is_retryable_exception(self, exception: Exception) -> bool:
        # return True if the exception should be retried
        return True

    def to_retryable_error(self, exception: Exception) -> RetryableError:
        # convert the exception to a retryable error
        return RetryableError(exception)


llm_with_custom_rate_limit_handler = CustomLLM(
    "", rate_limit_handler=CustomRateLimitHandler()
)
result: LLMResponse = llm_with_custom_rate_limit_handler.invoke(
    [{"role": "user", "content": "text"}]
)
print(result.content)
