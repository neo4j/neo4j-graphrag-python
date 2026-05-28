import warnings
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel

from neo4j_graphrag.types import LLMMessage as _LLMMessage


def __getattr__(name: str) -> Any:
    if name == "LLMMessage":
        warnings.warn(
            "LLMMessage has been moved to neo4j_graphrag.types. Please update your imports.",
            DeprecationWarning,
            stacklevel=2,
        )
        return _LLMMessage
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class LLMUsage(BaseModel):
    """Token usage statistics returned by an LLM call.

    Attributes:
        request_tokens (Optional[int]): Number of tokens in the prompt/request.
            ``None`` when not reported by the provider.
        response_tokens (Optional[int]): Number of tokens in the completion/response.
            ``None`` when not reported by the provider.
        total_tokens (Optional[int]): Total tokens consumed by the call.
            ``None`` when not reported by the provider.
    """

    request_tokens: Optional[int] = None
    response_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


class LLMResponse(BaseModel):
    """Response returned by an LLM invocation.

    Attributes:
        content (str): The text content of the LLM response.
        usage (Optional[LLMUsage]): Token usage statistics for the call, if provided by the LLM.
    """

    content: str
    usage: Optional[LLMUsage] = None


class BaseMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class UserMessage(BaseMessage):
    role: Literal["user"] = "user"


class SystemMessage(BaseMessage):
    role: Literal["system"] = "system"


class MessageList(BaseModel):
    messages: list[BaseMessage]


class ToolCall(BaseModel):
    """A tool call made by an LLM."""

    name: str
    arguments: Dict[str, Any]


class ToolCallResponse(BaseModel):
    """Response from an LLM containing tool calls."""

    tool_calls: List[ToolCall]
    content: Optional[str] = None
