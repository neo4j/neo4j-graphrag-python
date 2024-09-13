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
import re
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import ValidationError, validate_call

from neo4j_graphrag.exceptions import LLMGenerationError
from neo4j_graphrag.experimental.components.pdf_loader import DocumentInfo
from neo4j_graphrag.experimental.components.schema import SchemaConfig
from neo4j_graphrag.experimental.components.types import (
    Neo4jGraph,
    Neo4jNode,
    Neo4jRelationship,
    TextChunk,
    TextChunks,
)
from neo4j_graphrag.experimental.pipeline.component import Component
from neo4j_graphrag.generation.prompts import ERExtractionTemplate, PromptTemplate
from neo4j_graphrag.llm import LLMInterface

logger = logging.getLogger(__name__)


class OnError(enum.Enum):
    RAISE = "RAISE"
    IGNORE = "CONTINUE"


CHUNK_NODE_LABEL = "Chunk"
DOCUMENT_NODE_LABEL = "Document"
NEXT_CHUNK_RELATIONSHIP_TYPE = "NEXT_CHUNK"
NODE_TO_CHUNK_RELATIONSHIP_TYPE = "FROM_CHUNK"
CHUNK_TO_DOCUMENT_RELATIONSHIP_TYPE = "FROM_DOCUMENT"


def balance_curly_braces(json_string: str) -> str:
    """
    Balances curly braces `{}` in a JSON string. This function ensures that every opening brace has a corresponding
    closing brace, but only when they are not part of a string value. If there are unbalanced closing braces,
    they are ignored. If there are missing closing braces, they are appended at the end of the string.

    Args:
        json_string (str): A potentially malformed JSON string with unbalanced curly braces.

    Returns:
        str: A JSON string with balanced curly braces.
    """
    stack = []
    fixed_json = []
    in_string = False
    escape = False

    for char in json_string:
        if char == '"' and not escape:
            in_string = not in_string
        elif char == "\\" and in_string:
            escape = not escape
            fixed_json.append(char)
            continue
        else:
            escape = False

        if not in_string:
            if char == "{":
                stack.append(char)
                fixed_json.append(char)
            elif char == "}" and stack and stack[-1] == "{":
                stack.pop()
                fixed_json.append(char)
            elif char == "}" and (not stack or stack[-1] != "{"):
                continue
            else:
                fixed_json.append(char)
        else:
            fixed_json.append(char)

    # If stack is not empty, add missing closing braces
    while stack:
        stack.pop()
        fixed_json.append("}")

    return "".join(fixed_json)


def fix_invalid_json(invalid_json_string: str) -> str:
    # Fix missing quotes around field names
    invalid_json_string = re.sub(
        r"([{,]\s*)(\w+)(\s*:)", r'\1"\2"\3', invalid_json_string
    )

    # Fix missing quotes around string values, correctly ignoring null, true, false, and numeric values
    invalid_json_string = re.sub(
        r"(?<=:\s)(?!(null|true|false|\d+\.?\d*))([a-zA-Z_][a-zA-Z0-9_]*)\s*(?=[,}])",
        r'"\2"',
        invalid_json_string,
    )

    # Correct the specific issue: remove trailing commas within arrays or objects before closing braces or brackets
    invalid_json_string = re.sub(r",\s*(?=[}\]])", "", invalid_json_string)

    # Normalize excessive curly braces
    invalid_json_string = re.sub(r"{{+", "{", invalid_json_string)
    invalid_json_string = re.sub(r"}}+", "}", invalid_json_string)

    # Balance curly braces
    return balance_curly_braces(invalid_json_string)


class LexicalGraphBuilder:
    """A helper class to encompass useful methods to build the lexical graph"""

    @staticmethod
    def create_next_chunk_relationship(
        previous_chunk_id: str, chunk_id: str
    ) -> Neo4jRelationship:
        """Create relationship between a chunk and the next one"""
        return Neo4jRelationship(
            type=NEXT_CHUNK_RELATIONSHIP_TYPE,
            start_node_id=previous_chunk_id,
            end_node_id=chunk_id,
        )

    @staticmethod
    def create_chunk_node(chunk: TextChunk, chunk_id: str) -> Neo4jNode:
        """Create chunk node with properties 'text', 'index' and any 'metadata' added during
        the process. Special case for the potential chunk embedding property that
        gets added as an embedding_property"""
        chunk_properties: Dict[str, Any] = {
            "text": chunk.text,
            "index": chunk.index,
        }
        embedding_properties = {}
        if chunk.metadata:
            if "embedding" in chunk.metadata:
                embedding_properties["embedding"] = chunk.metadata.pop("embedding")
            chunk_properties.update(chunk.metadata)
        return Neo4jNode(
            id=chunk_id,
            label=CHUNK_NODE_LABEL,
            properties=chunk_properties,
            embedding_properties=embedding_properties,
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

    @staticmethod
    def create_document_node(document_info: DocumentInfo) -> Neo4jNode:
        """Create a Document node with 'path' property. Any document metadata is also
        added as a node property.
        """
        document_metadata = document_info.metadata or {}
        return Neo4jNode(
            id=document_info.path,
            label=DOCUMENT_NODE_LABEL,
            properties={
                "path": document_info.path,
                **document_metadata,
            },
        )

    @staticmethod
    def create_chunk_to_document_rel(
        chunk_id: str, document_id: str
    ) -> Neo4jRelationship:
        """Create the relationship between a chunk and the document it belongs to."""
        return Neo4jRelationship(
            start_node_id=chunk_id,
            end_node_id=document_id,
            type=CHUNK_TO_DOCUMENT_RELATIONSHIP_TYPE,
        )

    async def process_chunk(
        self,
        chunk_graph: Neo4jGraph,
        chunk: TextChunk,
        id_prefix: str,
        document_id: Optional[str] = None,
    ) -> None:
        """Add chunks and relationships between them (NEXT_CHUNK) and between
        chunks and extracted entities from that chunk.
        Updates `chunk_graph` in place.
        """
        chunk_id = f"{id_prefix}:{chunk.index}"
        if document_id:
            chunk_to_doc_rel = self.create_chunk_to_document_rel(chunk_id, document_id)
            chunk_graph.relationships.append(chunk_to_doc_rel)
        chunk_node = self.create_chunk_node(chunk, chunk_id)
        chunk_graph.nodes.append(chunk_node)
        if chunk.index > 0:
            previous_chunk_id = f"{id_prefix}:{chunk.index - 1}"
            next_chunk_rel = self.create_next_chunk_relationship(
                previous_chunk_id, chunk_id
            )
            chunk_graph.relationships.append(next_chunk_rel)
        for node in chunk_graph.nodes:
            if node.label in (CHUNK_NODE_LABEL, DOCUMENT_NODE_LABEL):
                continue
            node_to_chunk_rel = self.create_node_to_chunk_rel(node, chunk_id)
            chunk_graph.relationships.append(node_to_chunk_rel)


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
        self.on_error = on_error
        self.create_lexical_graph = create_lexical_graph

    @abc.abstractmethod
    async def run(
        self,
        chunks: TextChunks,
        document_info: Optional[DocumentInfo] = None,
        **kwargs: Any,
    ) -> Neo4jGraph:
        pass

    def update_ids(
        self, graph: Neo4jGraph, chunk_index: int, run_id: str
    ) -> Neo4jGraph:
        """Make node IDs unique across chunks and pipeline runs by
        prefixing them with a custom prefix (set in the run method)
        and chunk index."""
        prefix = f"{run_id}:{chunk_index}"
        for node in graph.nodes:
            node.id = f"{prefix}:{node.id}"
            if node.properties is None:
                node.properties = {}
            node.properties.update({"chunk_index": chunk_index})
        for rel in graph.relationships:
            rel.start_node_id = f"{prefix}:{rel.start_node_id}"
            rel.end_node_id = f"{prefix}:{rel.end_node_id}"
        return graph


class LLMEntityRelationExtractor(EntityRelationExtractor):
    """
    Extracts a knowledge graph from a series of text chunks using a large language model.

    Args:
        llm (LLMInterface): The language model to use for extraction.
        prompt_template (ERExtractionTemplate | str): A custom prompt template to use for extraction.
        create_lexical_graph (bool): Whether to include the text chunks in the graph in addition to the extracted entities and relations. Defaults to True.
        on_error (OnError): What to do when an error occurs during extraction. Defaults to raising an error.
        max_concurrency (int): The maximum number of concurrent tasks which can be used to make requests to the LLM.

    Example:

    .. code-block:: python

        from neo4j_graphrag.experimental.components.entity_relation_extractor import LLMEntityRelationExtractor
        from neo4j_graphrag.llm import OpenAILLM
        from neo4j_graphrag.experimental.pipeline import Pipeline

        llm = OpenAILLM(model_name="gpt-4o", model_params={"temperature": 0, "response_format": {"type": "object"}})

        extractor = LLMEntityRelationExtractor(llm=llm)
        pipe = Pipeline()
        pipe.add_component(extractor, "extractor")

    """

    def __init__(
        self,
        llm: LLMInterface,
        prompt_template: ERExtractionTemplate | str = ERExtractionTemplate(),
        create_lexical_graph: bool = True,
        on_error: OnError = OnError.RAISE,
        max_concurrency: int = 5,
    ) -> None:
        super().__init__(on_error=on_error, create_lexical_graph=create_lexical_graph)
        self.llm = llm  # with response_format={ "type": "json_object" },
        self.max_concurrency = max_concurrency
        if isinstance(prompt_template, str):
            template = PromptTemplate(prompt_template, expected_inputs=[])
        else:
            template = prompt_template
        self.prompt_template = template

    async def extract_for_chunk(
        self, schema: SchemaConfig, examples: str, chunk: TextChunk
    ) -> Neo4jGraph:
        """Run entity extraction for a given text chunk."""
        prompt = self.prompt_template.format(
            text=chunk.text, schema=schema.model_dump(), examples=examples
        )
        llm_result = await self.llm.ainvoke(prompt)
        try:
            result = json.loads(llm_result.content)
        except json.JSONDecodeError:
            logger.warning(
                f"LLM response is not valid JSON {llm_result.content} for chunk_index={chunk.index}. Trying to fix it."
            )
            fixed_content = fix_invalid_json(llm_result.content)
            try:
                result = json.loads(fixed_content)
            except json.JSONDecodeError as e:
                if self.on_error == OnError.RAISE:
                    raise LLMGenerationError(
                        f"LLM response is not valid JSON {fixed_content}: {e}"
                    )
                else:
                    logger.error(
                        f"LLM response is not valid JSON {llm_result.content} for chunk_index={chunk.index}"
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
                    f"LLM response has improper format {result} for chunk_index={chunk.index}"
                )
            chunk_graph = Neo4jGraph()
        return chunk_graph

    async def post_process_chunk(
        self,
        chunk_graph: Neo4jGraph,
        chunk: TextChunk,
        run_id: str,
        lexical_graph_builder: Optional[LexicalGraphBuilder] = None,
        document_id: Optional[str] = None,
    ) -> None:
        """Perform post-processing after entity and relation extraction:
        - Update node IDs to make them unique across chunks
        - Build the lexical graph if requested
        """
        self.update_ids(chunk_graph, chunk.index, run_id)
        if lexical_graph_builder:
            await lexical_graph_builder.process_chunk(
                chunk_graph, chunk, run_id, document_id=document_id
            )

    def combine_chunk_graphs(
        self, lexical_graph: Neo4jGraph, chunk_graphs: List[Neo4jGraph]
    ) -> Neo4jGraph:
        """Combine sub-graphs obtained for each chunk into a single Neo4jGraph object"""
        graph = lexical_graph.model_copy(deep=True)
        for chunk_graph in chunk_graphs:
            graph.nodes.extend(chunk_graph.nodes)
            graph.relationships.extend(chunk_graph.relationships)
        return graph

    async def run_for_chunk(
        self,
        sem: asyncio.Semaphore,
        run_id: str,
        chunk: TextChunk,
        schema: SchemaConfig,
        examples: str,
        lexical_graph_builder: Optional[LexicalGraphBuilder] = None,
        document_id: Optional[str] = None,
    ) -> Neo4jGraph:
        """Run extraction and post processing for a single chunk"""
        async with sem:
            chunk_graph = await self.extract_for_chunk(schema, examples, chunk)
            await self.post_process_chunk(
                chunk_graph, chunk, run_id, lexical_graph_builder, document_id
            )
            return chunk_graph

    @validate_call
    async def run(
        self,
        chunks: TextChunks,
        document_info: Optional[DocumentInfo] = None,
        schema: Union[SchemaConfig, None] = None,
        examples: str = "",
        **kwargs: Any,
    ) -> Neo4jGraph:
        """Perform entity and relation extraction for all chunks in a list."""
        lexical_graph_builder = None
        document_id = None
        nodes = []
        if self.create_lexical_graph:
            lexical_graph_builder = LexicalGraphBuilder()
            if document_info is None:
                warnings.warn(
                    "No document metadata provided, the document node won't be created in the lexical graph"
                )
            else:
                document_node = lexical_graph_builder.create_document_node(
                    document_info
                )
                nodes.append(document_node)
                document_id = document_node.id
        lexical_graph = Neo4jGraph(nodes=nodes, relationships=[])
        schema = schema or SchemaConfig(entities={}, relations={}, potential_schema=[])
        examples = examples or ""
        run_id = str(datetime.now().timestamp())
        sem = asyncio.Semaphore(self.max_concurrency)
        tasks = [
            self.run_for_chunk(
                sem, run_id, chunk, schema, examples, lexical_graph_builder, document_id
            )
            for chunk in chunks.chunks
        ]
        chunk_graphs: list[Neo4jGraph] = list(await asyncio.gather(*tasks))
        graph = self.combine_chunk_graphs(lexical_graph, chunk_graphs)
        logger.debug(f"{self.__class__.__name__}: {graph}")
        return graph
