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

from typing import Any, List, Optional, Union

import neo4j
from pydantic import BaseModel, ConfigDict, Field

from neo4j_graphrag.embeddings import Embedder
from neo4j_graphrag.experimental.components.embedder import TextChunkEmbedder
from neo4j_graphrag.experimental.components.entity_relation_extractor import (
    LLMEntityRelationExtractor,
    OnError,
)
from neo4j_graphrag.experimental.components.kg_writer import Neo4jWriter
from neo4j_graphrag.experimental.components.pdf_loader import PdfLoader
from neo4j_graphrag.experimental.components.resolver import (
    SinglePropertyExactMatchResolver,
)
from neo4j_graphrag.experimental.components.schema import (
    SchemaBuilder,
    SchemaEntity,
    SchemaRelation,
)
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import (
    FixedSizeSplitter,
)
from neo4j_graphrag.experimental.pipeline.exceptions import PipelineDefinitionError
from neo4j_graphrag.experimental.pipeline.pipeline import Pipeline, PipelineResult
from neo4j_graphrag.generation.prompts import ERExtractionTemplate
from neo4j_graphrag.llm.base import LLMInterface


class SimpleKGPipelineConfig(BaseModel):
    llm: LLMInterface
    driver: neo4j.Driver
    from_pdf: bool
    embedder: Embedder
    entities: list[SchemaEntity] = Field(default_factory=list)
    relations: list[SchemaRelation] = Field(default_factory=list)
    potential_schema: list[tuple[str, str, str]] = Field(default_factory=list)
    pdf_loader: Any = None
    kg_writer: Any = None
    text_splitter: Any = None
    on_error: OnError = OnError.RAISE
    prompt_template: Union[ERExtractionTemplate, str] = ERExtractionTemplate()
    perform_entity_resolution: bool = True

    model_config = ConfigDict(arbitrary_types_allowed=True)


class SimpleKGPipeline:
    """
    A class to simplify the process of building a knowledge graph from text documents.
    It abstracts away the complexity of setting up the pipeline and its components.

    Args:
        llm (LLMInterface): An instance of an LLM to use for entity and relation extraction.
        driver (neo4j.Driver): A Neo4j driver instance for database connection.
        embedder (Embedder): An instance of an embedder used to generate chunk embeddings from text chunks.
        entities (Optional[List[str]]): A list of entity labels as strings.
        relations (Optional[List[str]]): A list of relation labels as strings.
        potential_schema (Optional[List[tuple]]): A list of potential schema relationships.
        from_pdf (bool): Determines whether to include the PdfLoader in the pipeline.
                         If True, expects `file_path` input in `run` methods.
                         If False, expects `text` input in `run` methods.
        text_splitter (Optional[Any]): A text splitter component. Defaults to FixedSizeSplitter().
        pdf_loader (Optional[Any]): A PDF loader component. Defaults to PdfLoader().
        kg_writer (Optional[Any]): A knowledge graph writer component. Defaults to Neo4jWriter().
        on_error (str): Error handling strategy. Defaults to "RAISE". Possible values: "RAISE" or "IGNORE".
        perform_entity_resolution (bool): Merge entities with same label and name. Default: True
        text_splitter (Optional[Any]): A text splitter component. Defaults to FixedSizeSplitter().
        prompt_template (str): A custom prompt template to use for extraction.
    """

    def __init__(
        self,
        llm: LLMInterface,
        driver: neo4j.Driver,
        embedder: Embedder,
        entities: Optional[List[str]] = None,
        relations: Optional[List[str]] = None,
        potential_schema: Optional[List[tuple[str, str, str]]] = None,
        from_pdf: bool = True,
        text_splitter: Optional[Any] = None,
        pdf_loader: Optional[Any] = None,
        kg_writer: Optional[Any] = None,
        on_error: str = "RAISE",
        prompt_template: Union[ERExtractionTemplate, str] = ERExtractionTemplate(),
        perform_entity_resolution: bool = True,
    ):
        self.entities = [SchemaEntity(label=label) for label in entities or []]
        self.relations = [SchemaRelation(label=label) for label in relations or []]
        self.potential_schema = potential_schema if potential_schema is not None else []

        try:
            on_error_enum = OnError(on_error)
        except ValueError:
            raise PipelineDefinitionError(
                f"Invalid value for on_error: {on_error}. Expected 'RAISE' or 'CONTINUE'."
            )

        config = SimpleKGPipelineConfig(
            llm=llm,
            driver=driver,
            entities=self.entities,
            relations=self.relations,
            potential_schema=self.potential_schema,
            from_pdf=from_pdf,
            pdf_loader=pdf_loader,
            kg_writer=kg_writer,
            text_splitter=text_splitter,
            on_error=on_error_enum,
            prompt_template=prompt_template,
            embedder=embedder,
            perform_entity_resolution=perform_entity_resolution,
        )

        self.from_pdf = config.from_pdf
        self.llm = config.llm
        self.driver = config.driver
        self.embedder = config.embedder
        self.text_splitter = config.text_splitter or FixedSizeSplitter()
        self.on_error = config.on_error
        self.pdf_loader = config.pdf_loader if pdf_loader is not None else PdfLoader()
        self.kg_writer = (
            config.kg_writer if kg_writer is not None else Neo4jWriter(driver)
        )
        self.prompt_template = config.prompt_template
        self.perform_entity_resolution = config.perform_entity_resolution

        self.pipeline = self._build_pipeline()

    def _build_pipeline(self) -> Pipeline:
        pipe = Pipeline()

        pipe.add_component(self.text_splitter, "splitter")
        pipe.add_component(SchemaBuilder(), "schema")
        pipe.add_component(
            LLMEntityRelationExtractor(
                llm=self.llm,
                on_error=self.on_error,
                prompt_template=self.prompt_template,
            ),
            "extractor",
        )
        pipe.add_component(TextChunkEmbedder(embedder=self.embedder), "chunk_embedder")
        pipe.add_component(self.kg_writer, "writer")

        if self.from_pdf:
            pipe.add_component(self.pdf_loader, "pdf_loader")

            pipe.connect(
                "pdf_loader",
                "splitter",
                input_config={"text": "pdf_loader.text"},
            )

            pipe.connect(
                "schema",
                "extractor",
                input_config={
                    "schema": "schema",
                    "document_info": "pdf_loader.document_info",
                },
            )
        else:
            pipe.connect(
                "schema",
                "extractor",
                input_config={
                    "schema": "schema",
                },
            )

        pipe.connect(
            "splitter", "chunk_embedder", input_config={"text_chunks": "splitter"}
        )

        pipe.connect(
            "chunk_embedder", "extractor", input_config={"chunks": "chunk_embedder"}
        )

        # Connect extractor to writer
        pipe.connect(
            "extractor",
            "writer",
            input_config={"graph": "extractor"},
        )

        if self.perform_entity_resolution:
            pipe.add_component(
                SinglePropertyExactMatchResolver(self.driver), "resolver"
            )
            pipe.connect("writer", "resolver", {})

        return pipe

    async def run_async(
        self, file_path: Optional[str] = None, text: Optional[str] = None
    ) -> PipelineResult:
        """
        Asynchronously runs the knowledge graph building process.

        Args:
            file_path (Optional[str]): The path to the PDF file to process. Required if `from_pdf` is True.
            text (Optional[str]): The text content to process. Required if `from_pdf` is False.

        Returns:
            PipelineResult: The result of the pipeline execution.
        """
        pipe_inputs = self._prepare_inputs(file_path=file_path, text=text)
        return await self.pipeline.run(pipe_inputs)

    def _prepare_inputs(
        self, file_path: Optional[str], text: Optional[str]
    ) -> dict[str, Any]:
        if self.from_pdf:
            if file_path is None or text is not None:
                raise PipelineDefinitionError(
                    "Expected 'file_path' argument when 'from_pdf' is True."
                )
        else:
            if text is None or file_path is not None:
                raise PipelineDefinitionError(
                    "Expected 'text' argument when 'from_pdf' is False."
                )

        pipe_inputs: dict[str, Any] = {
            "schema": {
                "entities": self.entities,
                "relations": self.relations,
                "potential_schema": self.potential_schema,
            },
        }

        if self.from_pdf:
            pipe_inputs["pdf_loader"] = {"filepath": file_path}
        else:
            pipe_inputs["splitter"] = {"text": text}

        return pipe_inputs
