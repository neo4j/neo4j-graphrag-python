from pydantic import BaseModel
from typing import Literal


class LLMResponse(BaseModel):
    content: str


class BaseMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class UserMessage(BaseMessage):
    role: Literal["user"] = "user"

class SystemMessage(BaseMessage):
    role: Literal["system"] = "system"


class MessageList(BaseModel):
    messages: list[BaseMessage]
