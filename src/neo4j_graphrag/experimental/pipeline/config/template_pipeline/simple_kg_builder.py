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
from typing import Any, ClassVar, Literal, Optional, Sequence, Union

from pydantic import ConfigDict

from neo4j_graphrag.experimental.components.embedder import TextChunkEmbedder
from neo4j_graphrag.experimental.components.entity_relation_extractor import (
    EntityRelationExtractor,
    LLMEntityRelationExtractor,
    OnError,
)
from neo4j_graphrag.experimental.components.kg_writer import KGWriter, Neo4jWriter
from neo4j_graphrag.experimental.components.pdf_loader import PdfLoader
from neo4j_graphrag.experimental.components.resolver import (
    EntityResolver,
    SinglePropertyExactMatchResolver,
)
from neo4j_graphrag.experimental.components.schema import (
    SchemaBuilder,
    SchemaEntity,
    SchemaRelation,
)
from neo4j_graphrag.experimental.components.text_splitters.base import TextSplitter
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import (
    FixedSizeSplitter,
)
from neo4j_graphrag.experimental.components.types import LexicalGraphConfig
from neo4j_graphrag.experimental.pipeline.config.object_config import ComponentConfig
from neo4j_graphrag.experimental.pipeline.config.template_pipeline.base import (
    TemplatePipelineConfig,
)
from neo4j_graphrag.experimental.pipeline.config.types import PipelineType
from neo4j_graphrag.experimental.pipeline.exceptions import PipelineDefinitionError
from neo4j_graphrag.experimental.pipeline.types import (
    ConnectionDefinition,
    EntityInputType,
    RelationInputType,
)
from neo4j_graphrag.generation.prompts import ERExtractionTemplate


class SimpleKGPipelineConfig(TemplatePipelineConfig):
    COMPONENTS: ClassVar[list[str]] = [
        "pdf_loader",
        "splitter",
        "chunk_embedder",
        "schema",
        "extractor",
        "writer",
        "resolver",
    ]

    template_: Literal[PipelineType.SIMPLE_KG_PIPELINE] = (
        PipelineType.SIMPLE_KG_PIPELINE
    )

    from_pdf: bool = False
    entities: Sequence[EntityInputType] = []
    relations: Sequence[RelationInputType] = []
    potential_schema: Optional[list[tuple[str, str, str]]] = None
    on_error: OnError = OnError.IGNORE
    prompt_template: Union[ERExtractionTemplate, str] = ERExtractionTemplate()
    perform_entity_resolution: bool = True
    lexical_graph_config: Optional[LexicalGraphConfig] = None
    neo4j_database: Optional[str] = None

    pdf_loader: Optional[ComponentConfig] = None
    kg_writer: Optional[ComponentConfig] = None
    text_splitter: Optional[ComponentConfig] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_pdf_loader(self) -> Optional[PdfLoader]:
        if not self.from_pdf:
            return None
        if self.pdf_loader:
            return self.pdf_loader.parse(self._global_data)  # type: ignore
        return PdfLoader()

    def _get_splitter(self) -> TextSplitter:
        if self.text_splitter:
            return self.text_splitter.parse(self._global_data)  # type: ignore
        return FixedSizeSplitter()

    def _get_chunk_embedder(self) -> TextChunkEmbedder:
        return TextChunkEmbedder(embedder=self.get_default_embedder())

    def _get_schema(self) -> SchemaBuilder:
        return SchemaBuilder()

    def _get_run_params_for_schema(self) -> dict[str, Any]:
        return {
            "entities": [SchemaEntity.from_text_or_dict(e) for e in self.entities],
            "relations": [SchemaRelation.from_text_or_dict(r) for r in self.relations],
            "potential_schema": self.potential_schema,
        }

    def _get_extractor(self) -> EntityRelationExtractor:
        return LLMEntityRelationExtractor(
            llm=self.get_default_llm(),
            prompt_template=self.prompt_template,
            on_error=self.on_error,
        )

    def _get_writer(self) -> KGWriter:
        if self.kg_writer:
            return self.kg_writer.parse(self._global_data)  # type: ignore
        return Neo4jWriter(driver=self.get_default_neo4j_driver())

    def _get_resolver(self) -> Optional[EntityResolver]:
        if not self.perform_entity_resolution:
            return None
        return SinglePropertyExactMatchResolver(
            driver=self.get_default_neo4j_driver(),
        )

    def _get_connections(self) -> list[ConnectionDefinition]:
        connections = []
        if self.from_pdf:
            connections.append(
                ConnectionDefinition(
                    start="pdf_loader",
                    end="splitter",
                    input_config={"text": "pdf_loader.text"},
                )
            )
            connections.append(
                ConnectionDefinition(
                    start="schema",
                    end="extractor",
                    input_config={
                        "schema": "schema",
                        "document_info": "pdf_loader.document_info",
                    },
                )
            )
        else:
            connections.append(
                ConnectionDefinition(
                    start="schema",
                    end="extractor",
                    input_config={
                        "schema": "schema",
                    },
                )
            )
        connections.append(
            ConnectionDefinition(
                start="splitter",
                end="chunk_embedder",
                input_config={
                    "text_chunks": "splitter",
                },
            )
        )
        connections.append(
            ConnectionDefinition(
                start="chunk_embedder",
                end="extractor",
                input_config={
                    "chunks": "chunk_embedder",
                },
            )
        )
        connections.append(
            ConnectionDefinition(
                start="extractor",
                end="writer",
                input_config={
                    "graph": "extractor",
                },
            )
        )

        if self.perform_entity_resolution:
            connections.append(
                ConnectionDefinition(
                    start="writer",
                    end="resolver",
                    input_config={},
                )
            )

        return connections

    def get_run_params(self, user_input: dict[str, Any]) -> dict[str, Any]:
        run_params = {}
        if self.lexical_graph_config:
            run_params["extractor"] = {
                "lexical_graph_config": self.lexical_graph_config
            }
        text = user_input.get("text")
        file_path = user_input.get("file_path")
        if not ((text is None) ^ (file_path is None)):
            # exactly one of text or user_input must be set
            raise PipelineDefinitionError(
                "Use either 'text' (when from_pdf=False) or 'file_path' (when from_pdf=True) argument."
            )
        if self.from_pdf:
            if not file_path:
                raise PipelineDefinitionError(
                    "Expected 'file_path' argument when 'from_pdf' is True."
                )
            run_params["pdf_loader"] = {"filepath": file_path}
        else:
            if not text:
                raise PipelineDefinitionError(
                    "Expected 'text' argument when 'from_pdf' is False."
                )
            run_params["splitter"] = {"text": text}
        return run_params
