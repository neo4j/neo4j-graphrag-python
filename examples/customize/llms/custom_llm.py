import random
import string
from typing import Any, Awaitable, Callable, List, Optional, Type, TypeVar, Union

from pydantic import BaseModel

from neo4j_graphrag.llm import BaseLLM, LLMResponse
from neo4j_graphrag.utils.rate_limit import RateLimitHandler
from neo4j_graphrag.types import LLMMessage
from neo4j_graphrag.exceptions import RetryableError


class CustomLLM(BaseLLM):
    def __init__(
        self, model_name: str, system_instruction: Optional[str] = None, **kwargs: Any
    ):
        super().__init__(model_name, **kwargs)

    # Build a provider-specific request from the common message/response-format input.
    # Raise LLMGenerationError here for build-time failures (validation, schema errors).
    def _build_request(
        self,
        messages: List[LLMMessage],
        *,
        response_format: Optional[Union[Type[BaseModel], dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> str:
        return messages[-1]["content"]

    # Parse the raw response from _call_sync/_call_async into the common LLMResponse shape.
    def _parse_response(self, raw_response: Any) -> LLMResponse:
        return LLMResponse(content=raw_response)

    # Rate limiting (self._rate_limit_handler) is applied automatically by BaseLLM
    # around _call_sync/_call_async, so no decorator is needed here.
    def _call_sync(self, request: str) -> str:
        return (
            self.model_name + ": " + "".join(random.choices(string.ascii_letters, k=30))
        )

    async def _call_async(self, request: str) -> str:
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
