from pydantic import BaseModel


class LLMResponse(BaseModel):
    content: str
