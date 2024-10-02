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

import asyncio
from typing import Any, List, Optional

import neo4j
from pydantic import Field, BaseModel, model_validator, ConfigDict

from neo4j_graphrag.experimental.components.entity_relation_extractor import (
    LLMEntityRelationExtractor,
    OnError,
)
from neo4j_graphrag.experimental.components.kg_writer import Neo4jWriter
from neo4j_graphrag.experimental.components.pdf_loader import PdfLoader
from neo4j_graphrag.experimental.components.schema import (
    SchemaBuilder,
    SchemaEntity,
    SchemaRelation,
)
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import (
    FixedSizeSplitter,
)
from neo4j_graphrag.experimental.pipeline.pipeline import Pipeline, PipelineResult
from neo4j_graphrag.llm.base import LLMInterface


class KnowledgeGraphBuilderConfig(BaseModel):
    llm: LLMInterface
    driver: neo4j.Driver
    file_path: Optional[str] = None
    text: Optional[str] = None
    entities: list[SchemaEntity] = Field(default_factory=list)
    relations: list[SchemaRelation] = Field(default_factory=list)
    potential_schema: list[tuple] = Field(default_factory=list)
    pdf_loader: Any = None
    kg_writer: Any = None
    text_splitter: Any = None
    on_error: OnError = OnError.RAISE

    @model_validator(mode="before")
    def check_input_source(cls, values):
        file_path = values.get("file_path")
        text = values.get("text")
        if (file_path is None and text is None) or (
            file_path is not None and text is not None
        ):
            raise ValueError("Exactly one of 'file_path' or 'text' must be provided.")
        return values

    model_config = ConfigDict(arbitrary_types_allowed=True)


class KnowledgeGraphBuilder:
    """
    A class to simplify the process of building a knowledge graph from text documents.
    It abstracts away the complexity of setting up the pipeline and its components.

    Args:
        llm (LLMInterface): An instance of an LLM to use for entity and relation extraction.
        driver (neo4j.Driver): A Neo4j driver instance for database connection.
        file_path (Optional[str]): The path to the PDF file to process.
        text (Optional[str]): The text content to process.
        entities (Optional[List[str]]): A list of entity labels as strings.
        relations (Optional[List[str]]): A list of relation labels as strings.
        potential_schema (Optional[List[tuple]]): A list of potential schema relationships.
        text_splitter (Optional[Any]): A text splitter component. Defaults to FixedSizeSplitter().
        pdf_loader (Optional[Any]): A PDF loader component. Defaults to PdfLoader().
        kg_writer (Optional[Any]): A knowledge graph writer component. Defaults to Neo4jWriter().
        on_error (OnError): Error handling strategy. Defaults to OnError.RAISE.
    """

    def __init__(
        self,
        llm: LLMInterface,
        driver: neo4j.Driver,
        file_path: Optional[str] = None,
        text: Optional[str] = None,
        entities: Optional[List[str]] = None,
        relations: Optional[List[str]] = None,
        potential_schema: Optional[List[tuple]] = None,
        text_splitter: Optional[Any] = None,
        pdf_loader: Optional[Any] = None,
        kg_writer: Optional[Any] = None,
        on_error: OnError = OnError.RAISE,
    ):
        entities = [SchemaEntity(label=label) for label in entities or []]
        relations = [SchemaRelation(label=label) for label in relations or []]
        potential_schema = potential_schema if potential_schema is not None else []

        pdf_loader = pdf_loader if pdf_loader is not None else PdfLoader()
        kg_writer = kg_writer if kg_writer is not None else Neo4jWriter(driver)

        self.config = KnowledgeGraphBuilderConfig(
            llm=llm,
            driver=driver,
            file_path=file_path,
            text=text,
            entities=entities,
            relations=relations,
            potential_schema=potential_schema,
            text_splitter=text_splitter,
            pdf_loader=pdf_loader,
            kg_writer=kg_writer,
            on_error=on_error,
        )

        self.llm = self.config.llm
        self.driver = self.config.driver
        self.file_path = self.config.file_path
        self.text = self.config.text
        self.entities = self.config.entities
        self.relations = self.config.relations
        self.potential_schema = self.config.potential_schema
        self.text_splitter = self.config.text_splitter or FixedSizeSplitter()
        self.on_error = self.config.on_error
        self.pdf_loader = self.config.pdf_loader
        self.kg_writer = self.config.kg_writer

        self.pipeline = self._build_pipeline()

    def _build_pipeline(self) -> Pipeline:
        pipe = Pipeline()

        if self.file_path:
            pipe.add_component(self.pdf_loader, "loader")
            loader_inputs = {"filepath": self.file_path}
            document_info_input = "loader.document_info"
        else:
            loader_inputs = {}
            document_info_input = {"path": "direct_text_input"}

        pipe.add_component(
            self.text_splitter,
            "splitter",
        )

        pipe.add_component(SchemaBuilder(), "schema")
        pipe.add_component(
            LLMEntityRelationExtractor(llm=self.llm, on_error=self.on_error),
            "extractor",
        )
        pipe.add_component(self.kg_writer, "writer")

        if self.file_path:
            pipe.connect(
                "loader",
                "splitter",
                input_config={"text": "loader.text"},
            )
            self.pipe_inputs = {}
        else:
            self.pipe_inputs = {"splitter": {"text": self.text}}

        pipe.connect(
            "splitter",
            "extractor",
            input_config={"chunks": "splitter"},
        )
        pipe.connect(
            "schema",
            "extractor",
            input_config={
                "schema": "schema",
                "document_info": document_info_input,
            },
        )
        pipe.connect(
            "extractor",
            "writer",
            input_config={"graph": "extractor"},
        )

        self.pipe_inputs.update(
            {
                "schema": {
                    "entities": self.entities,
                    "relations": self.relations,
                    "potential_schema": self.potential_schema,
                },
            }
        )

        if self.file_path:
            self.pipe_inputs["loader"] = loader_inputs
        else:
            self.pipe_inputs["extractor"] = {"document_info": document_info_input}

        return pipe

    async def run_async(self) -> PipelineResult:
        """
        Asynchronously runs the knowledge graph building process.

        Returns:
            PipelineResult: The result of the pipeline execution.
        """
        return await self.pipeline.run(self.pipe_inputs)

    def run(self) -> PipelineResult:
        """
        Runs the knowledge graph building process.

        Returns:
            PipelineResult: The result of the pipeline execution.
        """
        return asyncio.run(self.run_async())
