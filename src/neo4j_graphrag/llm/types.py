from typing import Literal, TypedDict, Optional

from pydantic import BaseModel


class ToolCall(BaseModel):
    id: str
    type: Literal["function"]
    name: str
    arguments: str


class LLMResponse(BaseModel):
    content: str
    tool_calls: Optional[list[ToolCall]] = None


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
