import warnings
from typing import Any, Literal

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


class LLMResponse(BaseModel):
    content: str


class BaseMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class UserMessage(BaseMessage):
    role: Literal["user"] = "user"


class SystemMessage(BaseMessage):
    role: Literal["system"] = "system"


class MessageList(BaseModel):
    messages: list[BaseMessage]
