from typing import Literal, TypedDict, Optional, Any

from pydantic import BaseModel


class LLMResponse(BaseModel):
    content: str
    function_call: Optional[dict[str, Any]] = None


class LLMMessage(TypedDict):
    role: Literal["system", "user", "assistant"]
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
