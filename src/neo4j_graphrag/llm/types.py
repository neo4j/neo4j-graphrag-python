import warnings
from typing import Literal

from pydantic import BaseModel

from neo4j_graphrag.types import LLMMessage as _LLMMessage

warnings.warn(
    "LLMMessage has been moved to neo4j_graphrag.types. Please update your imports.",
    DeprecationWarning,
    stacklevel=2,
)
LLMMessage = _LLMMessage


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
