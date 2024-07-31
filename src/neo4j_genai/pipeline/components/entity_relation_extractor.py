from __future__ import annotations

import enum
import json
from typing import Any

from pydantic import BaseModel

from neo4j_genai.exceptions import LLMGenerationError
from neo4j_genai.generation.prompts import ERExtractionTemplate
from neo4j_genai.llm import LLMInterface
from neo4j_genai.pipeline.component import Component, DataModel


class EntityModel(BaseModel):
    id: str
    label: str
    properties: dict[str, Any]


class RelationModel(BaseModel):
    id: str
    label: str
    source_entity: str
    target_entity: str
    properties: dict[str, Any]


class EntityRelationGraphModel(BaseModel):
    entities: list[EntityModel]
    relations: list[RelationModel]


class ERResultModel(DataModel):
    result: list[EntityRelationGraphModel]


class EntityRelationExtractor(Component):
    async def run(self, chunks: list[str], **kwargs: Any) -> ERResultModel:
        # for each chunk, returns a dict with entities and relations keys
        return ERResultModel(result=[])


class OnError(enum.Enum):
    RAISE = "RAISE"
    IGNORE = "CONTINUE"


class LLMEntityRelationExtractor(EntityRelationExtractor):
    def __init__(
        self,
        llm: LLMInterface,
        prompt_template: ERExtractionTemplate = ERExtractionTemplate(),
        on_error: OnError = OnError.RAISE,
    ) -> None:
        self.llm = llm  # with response_format={ "type": "json_object" },
        self.prompt_template = prompt_template
        self.on_error = on_error

    # TODO: fix the type of "schema" and "examples"
    async def run(
        self,
        chunks: list[str],
        schema: BaseModel | dict[str, Any] | None = None,
        examples: Any = None,
        **kwargs: Any,
    ) -> ERResultModel:
        # TODO: deal with tools (for function calling)?
        chunk_results = []
        for index, chunk in enumerate(chunks):
            prompt = self.prompt_template.format(
                text=chunk, schema=schema, examples=examples
            )
            llm_result = self.llm.invoke(prompt)
            try:
                result = json.loads(llm_result.content)
            except json.JSONDecodeError:
                if self.on_error == OnError.RAISE:
                    raise LLMGenerationError(
                        f"LLM response is not valid JSON {llm_result.content}"
                    )
                result = {"entities": [], "relations": [], "error": True}
            chunk_results.append(result)
        return ERResultModel(result=chunk_results)


if __name__ == "__main__":
    from neo4j_genai.llm import OpenAILLM

    llm = OpenAILLM(
        model_name="gpt-4o", model_params={"response_format": {"type": "json_object"}}
    )
    extractor = LLMEntityRelationExtractor(llm)
    result = extractor.run(
        chunks=[
            "Emil Eifrem is the CEO of Neo4j.",
            "Mark is a Freemason",
            "Alice belongs to the Freemasonry organization",
        ],
        schema={
            "entities": [
                {"label": "Person", "properties": [{"name": "name", "type": "STRING"}]},
                {
                    "label": "Organization",
                    "properties": [{"name": "name", "type": "STRING"}],
                },
            ],
            "relations": [
                {
                    "label": "MEMBER_OF",
                    "source_node_type": "Person",
                    "target_node_type": "Organization",
                    "properties": [],
                },
            ],
        },
    )
    print(result)
