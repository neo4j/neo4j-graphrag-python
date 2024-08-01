#  Copyright (c) "Neo4j"
#  Neo4j Sweden AB [https://neo4j.com]
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#      https://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
from __future__ import annotations

import enum
import json
import logging
from typing import Any

from pydantic import BaseModel, ValidationError, validate_call

from neo4j_genai.components.types import Neo4jGraph, TextChunks
from neo4j_genai.exceptions import LLMGenerationError
from neo4j_genai.generation.prompts import ERExtractionTemplate, PromptTemplate
from neo4j_genai.llm import LLMInterface
from neo4j_genai.pipeline.component import Component

logger = logging.getLogger(__name__)


class EntityRelationExtractor(Component):
    async def run(self, chunks: TextChunks, **kwargs: Any) -> Neo4jGraph:
        # for each chunk, returns a dict with entities and relations keys
        return Neo4jGraph(nodes=[], relationships=[])


class OnError(enum.Enum):
    RAISE = "RAISE"
    IGNORE = "CONTINUE"


class LLMEntityRelationExtractor(EntityRelationExtractor):
    def __init__(
        self,
        llm: LLMInterface,
        prompt_template: ERExtractionTemplate | str = ERExtractionTemplate(),
        on_error: OnError = OnError.RAISE,
    ) -> None:
        self.llm = llm  # with response_format={ "type": "json_object" },
        if isinstance(prompt_template, str):
            template = PromptTemplate(prompt_template, expected_inputs=[])
        else:
            template = prompt_template
        self.prompt_template = template
        self.on_error = on_error

    def update_ids(self, chunk_index: int, graph: Neo4jGraph) -> Neo4jGraph:
        for node in graph.nodes:
            node.id = f"{chunk_index}:{node.id}"
        for rel in graph.relationships:
            rel.start_node_id = f"{chunk_index}:{rel.start_node_id}"
            rel.end_node_id = f"{chunk_index}:{rel.end_node_id}"
        return graph

    # TODO: fix the type of "schema" and "examples"
    @validate_call
    async def run(
        self,
        chunks: TextChunks,
        schema: BaseModel | dict[str, Any] | None = None,
        examples: Any = None,
        **kwargs: Any,
    ) -> Neo4jGraph:
        schema = schema or {}
        examples = examples or ""
        graph = Neo4jGraph()
        for index, chunk in enumerate(chunks.chunks):
            prompt = self.prompt_template.format(
                text=chunk.text, schema=schema, examples=examples
            )
            llm_result = self.llm.invoke(prompt)
            try:
                result = json.loads(llm_result.content)
            except json.JSONDecodeError:
                if self.on_error == OnError.RAISE:
                    logger.error(f"LLM response is not valid JSON {llm_result.content}")
                    raise LLMGenerationError(
                        f"LLM response is not valid JSON {llm_result.content}"
                    )
                result = {"nodes": [], "relationships": []}
            try:
                chunk_graph = Neo4jGraph(**result)
            except ValidationError:
                if self.on_error == OnError.RAISE:
                    logger.error(
                        f"LLM response has improper format {llm_result.content}"
                    )
                    raise LLMGenerationError(
                        f"LLM response has improper format {llm_result.content}"
                    )
                chunk_graph = Neo4jGraph()
            self.update_ids(index, chunk_graph)
            graph.nodes.extend(chunk_graph.nodes)
            graph.relationships.extend(chunk_graph.relationships)
        logger.debug(f"{self.__class__.__name__}: {graph}")
        return graph
