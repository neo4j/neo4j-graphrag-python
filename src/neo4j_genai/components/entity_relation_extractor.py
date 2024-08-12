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

import abc
import asyncio
import enum
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Union

from pydantic import ValidationError, validate_call

from neo4j_genai.components.schema import SchemaConfig
from neo4j_genai.components.types import (
    Neo4jGraph,
    Neo4jNode,
    Neo4jRelationship,
    TextChunk,
    TextChunks,
)
from neo4j_genai.exceptions import LLMGenerationError
from neo4j_genai.generation.prompts import ERExtractionTemplate, PromptTemplate
from neo4j_genai.llm import LLMInterface
from neo4j_genai.pipeline.component import Component

logger = logging.getLogger(__name__)


class OnError(enum.Enum):
    RAISE = "RAISE"
    IGNORE = "CONTINUE"


CHUNK_NODE_LABEL = "Chunk"
NEXT_CHUNK_RELATIONSHIP_TYPE = "NEXT_CHUNK"
NODE_TO_CHUNK_RELATIONSHIP_TYPE = "FROM_CHUNK"


class EntityRelationExtractor(Component, abc.ABC):
    """Abstract class for entity relation extraction components.

    Args:
        on_error (OnError): What to do when an error occurs during extraction. Defaults to raising an error.
        create_lexical_graph (bool): Whether to include the text chunks in the graph in addition to the extracted entities and relations. Defaults to True.
    """

    def __init__(
        self,
        *args: Any,
        on_error: OnError = OnError.IGNORE,
        create_lexical_graph: bool = True,
        **kwargs: Any,
    ) -> None:
        self.create_lexical_graph = create_lexical_graph
        self.on_error = on_error
        self._id_prefix = ""

    @abc.abstractmethod
    async def run(self, chunks: TextChunks, **kwargs: Any) -> Neo4jGraph:
        pass

    def update_ids(self, graph: Neo4jGraph, chunk_index: int) -> Neo4jGraph:
        """Make node IDs unique across chunks and pipeline runs by
        prefixing them with a custom prefix (set in the run method)
        and chunk index."""
        for node in graph.nodes:
            node.id = f"{self._id_prefix}:{chunk_index}:{node.id}"
            if node.properties is None:
                node.properties = {}
            node.properties.update({"chunk_index": chunk_index})
        for rel in graph.relationships:
            rel.start_node_id = f"{self._id_prefix}:{chunk_index}:{rel.start_node_id}"
            rel.end_node_id = f"{self._id_prefix}:{chunk_index}:{rel.end_node_id}"
        return graph

    def create_next_chunk_relationship(
        self, previous_chunk_id: str, chunk_id: str
    ) -> Neo4jRelationship:
        """Create relationship between a chunk and the next one"""
        return Neo4jRelationship(
            type=NEXT_CHUNK_RELATIONSHIP_TYPE,
            start_node_id=previous_chunk_id,
            end_node_id=chunk_id,
        )

    def create_chunk_node(self, chunk: TextChunk, chunk_id: str) -> Neo4jNode:
        """Create chunk node with properties 'text' and 'metadata' if metadata is defined."""
        chunk_properties: Dict[str, Any] = {
            "text": chunk.text,
        }
        if chunk.metadata:
            chunk_properties["metadata"] = chunk.metadata
        return Neo4jNode(
            id=chunk_id,
            label=CHUNK_NODE_LABEL,
            properties=chunk_properties,
        )

    def create_node_to_chunk_rel(
        self, node: Neo4jNode, chunk_id: str
    ) -> Neo4jRelationship:
        """Create relationship between a chunk and entities found in that chunk"""
        return Neo4jRelationship(
            start_node_id=node.id,
            end_node_id=chunk_id,
            type=NODE_TO_CHUNK_RELATIONSHIP_TYPE,
        )

    def build_lexical_graph(
        self, chunk_graph: Neo4jGraph, chunk_index: int, chunk: TextChunk
    ) -> Neo4jGraph:
        """Add chunks and relationships between them (NEXT_CHUNK) and between
        chunks and extracted entities from that chunk.
        """
        chunk_id = f"{self._id_prefix}:{chunk_index}"
        chunk_node = self.create_chunk_node(chunk, chunk_id)
        chunk_graph.nodes.append(chunk_node)
        if chunk_index > 0:
            previous_chunk_id = f"{self._id_prefix}:{chunk_index - 1}"
            next_chunk_rel = self.create_next_chunk_relationship(
                previous_chunk_id, chunk_id
            )
            chunk_graph.relationships.append(next_chunk_rel)
        for node in chunk_graph.nodes:
            if node.label == CHUNK_NODE_LABEL:
                continue
            node_to_chunk_rel = self.create_node_to_chunk_rel(node, chunk_id)
            chunk_graph.relationships.append(node_to_chunk_rel)
        return chunk_graph


class LLMEntityRelationExtractor(EntityRelationExtractor):
    """
    Extracts a knowledge graph from a series of text chunks using a large language model.

    Args:
        llm (LLMInterface): The language model to use for extraction.
        prompt_template (ERExtractionTemplate | str): A custom prompt template to use for extraction.
        create_lexical_graph (bool): Whether to include the text chunks in the graph in addition to the extracted entities and relations. Defaults to True.
        on_error (OnError): What to do when an error occurs during extraction. Defaults to raising an error.

    Example:

    .. code-block:: python

        from neo4j_genai.components.entity_relation_extractor import LLMEntityRelationExtractor
        from neo4j_genai.llm import OpenAILLM
        from neo4j_genai.pipeline import Pipeline

        llm = OpenAILLM(model_name="gpt-4o", model_params={"temperature": 0, "response_format": {"type": "object"}})

        extractor = LLMEntityRelationExtractor(llm=llm)
        pipe = Pipeline()
        pipe.add_component("extractor", extractor)

    """

    def __init__(
        self,
        llm: LLMInterface,
        prompt_template: ERExtractionTemplate | str = ERExtractionTemplate(),
        create_lexical_graph: bool = True,
        on_error: OnError = OnError.RAISE,
    ) -> None:
        super().__init__(on_error=on_error, create_lexical_graph=create_lexical_graph)
        self.llm = llm  # with response_format={ "type": "json_object" },
        if isinstance(prompt_template, str):
            template = PromptTemplate(prompt_template, expected_inputs=[])
        else:
            template = prompt_template
        self.prompt_template = template

    async def extract_for_chunk(
        self, schema: SchemaConfig, examples: str, chunk_index: int, chunk: TextChunk
    ) -> Neo4jGraph:
        """Run entity extraction for a given text chunk."""
        prompt = self.prompt_template.format(
            text=chunk.text, schema=schema.model_dump(), examples=examples
        )
        llm_result = self.llm.invoke(prompt)
        try:
            result = json.loads(llm_result.content)
        except json.JSONDecodeError:
            if self.on_error == OnError.RAISE:
                raise LLMGenerationError(
                    f"LLM response is not valid JSON {llm_result.content}"
                )
            else:
                logger.error(
                    f"LLM response is not valid JSON {llm_result.content} for chunk_index={chunk_index}"
                )
            result = {"nodes": [], "relationships": []}
        try:
            chunk_graph = Neo4jGraph(**result)
        except ValidationError as e:
            if self.on_error == OnError.RAISE:
                raise LLMGenerationError(
                    f"LLM response has improper format {result}: {e}"
                )
            else:
                logger.error(
                    f"LLM response has improper format {result} for chunk_index={chunk_index}"
                )
            chunk_graph = Neo4jGraph()
        return chunk_graph

    async def post_process_chunk(
        self, chunk_graph: Neo4jGraph, chunk_index: int, chunk: TextChunk
    ) -> None:
        """Perform post-processing after entity and relation extraction:
        - Update node IDs to make them unique across chunks
        - Build the lexical graph if requested
        """
        self.update_ids(chunk_graph, chunk_index)
        if self.create_lexical_graph:
            self.build_lexical_graph(chunk_graph, chunk_index, chunk)

    def combine_chunk_graphs(self, chunk_graphs: List[Neo4jGraph]) -> Neo4jGraph:
        """Combine sub-graphs obtained for each chunk into a single Neo4jGraph object"""
        graph = Neo4jGraph()
        for chunk_graph in chunk_graphs:
            graph.nodes.extend(chunk_graph.nodes)
            graph.relationships.extend(chunk_graph.relationships)
        return graph

    async def run_for_chunk(
        self, schema: SchemaConfig, examples: str, chunk_index: int, chunk: TextChunk
    ) -> Neo4jGraph:
        """Run extraction and post processing for a single chunk"""
        chunk_graph = await self.extract_for_chunk(schema, examples, chunk_index, chunk)
        await self.post_process_chunk(chunk_graph, chunk_index, chunk)
        return chunk_graph

    @validate_call
    async def run(
        self,
        chunks: TextChunks,
        schema: Union[SchemaConfig, None] = None,
        examples: str = "",
        **kwargs: Any,
    ) -> Neo4jGraph:
        """Perform entity and relation extraction for all chunks in a list."""
        schema = schema or SchemaConfig(entities={}, relations={}, potential_schema=[])
        examples = examples or ""
        self._id_prefix = str(datetime.now().timestamp())
        tasks = [
            self.run_for_chunk(schema, examples, chunk_index, chunk)
            for chunk_index, chunk in enumerate(chunks.chunks)
        ]
        chunk_graphs = await asyncio.gather(*tasks)
        graph = self.combine_chunk_graphs(chunk_graphs)
        logger.debug(f"{self.__class__.__name__}: {graph}")
        return graph
