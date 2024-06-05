from pydantic import BaseModel, ConfigDict
from neo4j_genai.types import RetrieverResult


class RagResultModel(BaseModel):
    answer: str
    retriever_result: RetrieverResult | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)
