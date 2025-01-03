from typing import Literal, TypedDict

from pydantic import BaseModel


class LLMResponse(BaseModel):
    content: str


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
