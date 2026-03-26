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

from collections import defaultdict
from pathlib import Path
from typing import (
    Any,
    ClassVar,
    Literal,
    Optional,
    Sequence,
    Union,
)
import warnings

from pydantic import ConfigDict, Field, field_validator, model_validator
from typing_extensions import Self

from neo4j_graphrag.experimental.components.embedder import TextChunkEmbedder
from neo4j_graphrag.experimental.components.entity_relation_extractor import (
    EntityRelationExtractor,
    LLMEntityRelationExtractor,
    OnError,
)
from neo4j_graphrag.experimental.components.graph_pruning import GraphPruning
from neo4j_graphrag.experimental.components.kg_writer import KGWriter, Neo4jWriter
from neo4j_graphrag.exceptions import UnsupportedDocumentFormatError
from neo4j_graphrag.experimental.components.data_loader import (
    DataLoader,
    MarkdownLoader,
    PdfLoader,
)
from neo4j_graphrag.experimental.components.resolver import (
    EntityResolver,
    SinglePropertyExactMatchResolver,
)
from neo4j_graphrag.experimental.components.schema import (
    SchemaBuilder,
    GraphSchema,
    SchemaFromTextExtractor,
    BaseSchemaBuilder,
)
from neo4j_graphrag.experimental.components.text_splitters.base import TextSplitter
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import (
    FixedSizeSplitter,
)
from neo4j_graphrag.experimental.components.types import (
    DocumentType,
    LexicalGraphConfig,
    LoadedDocument,
)
from neo4j_graphrag.experimental.pipeline.config.object_config import ComponentType
from neo4j_graphrag.experimental.pipeline.config.template_pipeline.base import (
    TemplatePipelineConfig,
)
from neo4j_graphrag.experimental.pipeline.config.types import PipelineType
from neo4j_graphrag.experimental.pipeline.exceptions import PipelineDefinitionError
from neo4j_graphrag.experimental.pipeline.types.definitions import ConnectionDefinition
from neo4j_graphrag.experimental.pipeline.types.schema import (
    EntityInputType,
    RelationInputType,
)
from neo4j_graphrag.generation.prompts import ERExtractionTemplate


class _DefaultPathDataLoader(DataLoader):
    """Default loader for ``SimpleKGPipeline`` that supports PDF and Markdown."""

    async def run(
        self,
        filepath: Union[str, Path],
        metadata: Optional[dict[str, str]] = None,
    ) -> LoadedDocument:
        path_str = str(filepath)
        suffix = Path(path_str).suffix.lower()
        if suffix == ".pdf":
            return await PdfLoader().run(filepath=path_str, metadata=metadata)
        if suffix in (".md", ".markdown"):
            return await MarkdownLoader().run(filepath=path_str, metadata=metadata)
        raise UnsupportedDocumentFormatError(
            f"Unsupported document format: {suffix!r}. "
            f"Supported: .pdf, .md, .markdown"
        )


class SimpleKGPipelineConfig(TemplatePipelineConfig):
    COMPONENTS: ClassVar[list[str]] = [
        "file_loader",
        "splitter",
        "chunk_embedder",
        "schema",
        "extractor",
        "pruner",
        "writer",
        "resolver",
    ]

    template_: Literal[PipelineType.SIMPLE_KG_PIPELINE] = (
        PipelineType.SIMPLE_KG_PIPELINE
    )

    from_file: bool = False
    from_pdf: Optional[bool] = Field(
        default=None,
        exclude=True,
        description="Deprecated. Use `from_file` instead.",
    )
    entities: Sequence[EntityInputType] = []
    relations: Sequence[RelationInputType] = []
    potential_schema: Optional[list[tuple[str, str, str]]] = None
    schema_: Optional[GraphSchema] = Field(default=None, alias="schema")
    on_error: OnError = OnError.IGNORE
    prompt_template: Union[ERExtractionTemplate, str] = ERExtractionTemplate()
    perform_entity_resolution: bool = True
    lexical_graph_config: Optional[LexicalGraphConfig] = None
    neo4j_database: Optional[str] = None

    file_loader: Optional[ComponentType] = None
    pdf_loader: Optional[ComponentType] = Field(
        default=None,
        exclude=True,
        description="Deprecated. Use `file_loader` instead.",
    )
    kg_writer: Optional[ComponentType] = None
    text_splitter: Optional[ComponentType] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def apply_deprecated_from_pdf_and_pdf_loader(self) -> Self:
        """Map legacy ``from_pdf`` / ``pdf_loader`` to ``from_file`` / ``file_loader``."""
        if self.from_pdf is not None:
            warnings.warn(
                "`from_pdf` is deprecated and will be removed in a future version; "
                "use `from_file` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.from_file = self.from_pdf
        if self.pdf_loader is not None:
            if self.file_loader is not None:
                raise ValueError(
                    "Pass only one of `file_loader` and `pdf_loader`; "
                    "`pdf_loader` is deprecated."
                )
            warnings.warn(
                "`pdf_loader` is deprecated and will be removed in a future version; "
                "use `file_loader` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.file_loader = self.pdf_loader
        return self

    @field_validator("schema_", mode="before")
    @classmethod
    def validate_schema_literal(cls, v: Any) -> Any:
        if v == "FREE":  # same as "empty" schema, no guiding schema
            return GraphSchema.create_empty()
        if v == "EXTRACTED":  # same as no schema, schema will be extracted by LLM
            return None
        return v

    @model_validator(mode="after")
    def handle_schema_precedence(self) -> Self:
        """Handle schema precedence and warnings"""
        self._process_schema_parameters()
        return self

    def _process_schema_parameters(self) -> None:
        """
        Process schema parameters and handle precedence between 'schema' parameter and individual components.
        Also logs warnings for deprecated usage.
        """
        # check if both schema and individual components are provided
        has_individual_schema_components = any(
            [self.entities, self.relations, self.potential_schema]
        )

        if has_individual_schema_components and self.schema_ is not None:
            warnings.warn(
                "Both 'schema' and individual schema components (entities, relations, potential_schema) "
                "were provided. The 'schema' parameter takes precedence. In the future, individual "
                "components will be removed. Please use only the 'schema' parameter.",
                DeprecationWarning,
                stacklevel=2,
            )

        elif has_individual_schema_components:
            warnings.warn(
                "The 'entities', 'relations', and 'potential_schema' parameters are deprecated "
                "and will be removed in a future version. "
                "Please use the 'schema' parameter instead.",
                DeprecationWarning,
                stacklevel=2,
            )

    def has_user_provided_schema(self) -> bool:
        """Check if the user has provided schema information"""
        return bool(
            self.entities
            or self.relations
            or self.potential_schema
            or self.schema_ is not None
        )

    def _get_file_loader(self) -> Optional[DataLoader]:
        if not self.from_file:
            return None
        if self.file_loader:
            return self.file_loader.parse(self._global_data)  # type: ignore
        return _DefaultPathDataLoader()

    def _get_run_params_for_file_loader(self) -> dict[str, Any]:
        if not self.from_file:
            return {}
        if self.file_loader:
            return self.file_loader.get_run_params(self._global_data)
        return {}

    def _get_splitter(self) -> TextSplitter:
        if self.text_splitter:
            return self.text_splitter.parse(self._global_data)  # type: ignore
        return FixedSizeSplitter()

    def _get_run_params_for_splitter(self) -> dict[str, Any]:
        if self.text_splitter:
            return self.text_splitter.get_run_params(self._global_data)
        return {}

    def _get_chunk_embedder(self) -> TextChunkEmbedder:
        return TextChunkEmbedder(embedder=self.get_default_embedder())

    def _get_schema(self) -> BaseSchemaBuilder:
        """
        Get the appropriate schema component based on configuration.
        Return SchemaFromTextExtractor for automatic extraction or SchemaBuilder for manual schema.
        """
        if not self.has_user_provided_schema():
            llm = self.get_default_llm()
            return SchemaFromTextExtractor(
                llm=llm,
                use_structured_output=llm.supports_structured_output,
            )
        return SchemaBuilder()

    def _process_schema_with_precedence(self) -> dict[str, Any]:
        """
        Process schema inputs according to precedence rules:
        1. If schema is provided as GraphSchema object, use it
        2. If schema is provided as dictionary, extract from it
        3. Otherwise, use individual schema components

        Returns:
            A dict representing the schema
        """
        if self.schema_ is not None:
            return self.schema_.model_dump()

        return dict(
            node_types=self.entities,
            relationship_types=self.relations,
            patterns=self.potential_schema,
        )

    def _get_run_params_for_schema(self) -> dict[str, Any]:
        if not self.has_user_provided_schema():
            # for automatic extraction, the text parameter is needed (will flow through the pipeline connections)
            return {}
        else:
            # process schema components according to precedence rules
            schema_dict = self._process_schema_with_precedence()
            return schema_dict

    def _get_extractor(self) -> EntityRelationExtractor:
        llm = self.get_default_llm()
        return LLMEntityRelationExtractor(
            llm=llm,
            prompt_template=self.prompt_template,
            on_error=self.on_error,
            use_structured_output=llm.supports_structured_output,
        )

    def _get_pruner(self) -> GraphPruning:
        return GraphPruning()

    def _get_writer(self) -> KGWriter:
        if self.kg_writer:
            return self.kg_writer.parse(self._global_data)  # type: ignore
        return Neo4jWriter(
            driver=self.get_default_neo4j_driver(),
            neo4j_database=self.neo4j_database,
        )

    def _get_run_params_for_writer(self) -> dict[str, Any]:
        if self.kg_writer:
            return self.kg_writer.get_run_params(self._global_data)
        return {}

    def _get_resolver(self) -> Optional[EntityResolver]:
        if not self.perform_entity_resolution:
            return None
        return SinglePropertyExactMatchResolver(
            driver=self.get_default_neo4j_driver(),
            neo4j_database=self.neo4j_database,
        )

    def _get_connections(self) -> list[ConnectionDefinition]:
        connections = []
        if self.from_file:
            connections.append(
                ConnectionDefinition(
                    start="file_loader",
                    end="splitter",
                    input_config={"text": "file_loader.text"},
                )
            )

            # handle automatic schema extraction
            if not self.has_user_provided_schema():
                connections.append(
                    ConnectionDefinition(
                        start="file_loader",
                        end="schema",
                        input_config={"text": "file_loader.text"},
                    )
                )

            connections.append(
                ConnectionDefinition(
                    start="schema",
                    end="extractor",
                    input_config={
                        "schema": "schema",
                        "document_info": "file_loader.document_info",
                    },
                )
            )
        else:
            connections.append(
                ConnectionDefinition(
                    start="schema",
                    end="extractor",
                    input_config={"schema": "schema"},
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
                end="pruner",
                input_config={
                    "graph": "extractor",
                    "schema": "schema",
                },
            )
        )
        connections.append(
            ConnectionDefinition(
                start="pruner",
                end="writer",
                input_config={
                    "graph": "pruner.graph",
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
        text = user_input.get("text")
        file_path = user_input.get("file_path")
        if text is None and file_path is None:
            # user must provide either text or file_path or both
            raise PipelineDefinitionError(
                "At least one of `text` (when from_file=False) or `file_path` (when from_file=True) argument must be provided."
            )
        run_params: dict[str, dict[str, Any]] = defaultdict(dict)
        if self.lexical_graph_config:
            run_params["extractor"]["lexical_graph_config"] = self.lexical_graph_config
            run_params["writer"]["lexical_graph_config"] = self.lexical_graph_config
            run_params["pruner"]["lexical_graph_config"] = self.lexical_graph_config
        if self.from_file:
            if not file_path:
                raise PipelineDefinitionError(
                    "Expected 'file_path' to a PDF or Markdown file when 'from_file' is True."
                )
            run_params["file_loader"]["filepath"] = file_path
            run_params["file_loader"]["metadata"] = user_input.get("document_metadata")
        else:
            if not text:
                raise PipelineDefinitionError(
                    "Expected 'text' argument when 'from_file' is False."
                )
            run_params["splitter"]["text"] = text
            # Add full text to schema component for automatic schema extraction
            if not self.has_user_provided_schema():
                run_params["schema"]["text"] = text
            run_params["extractor"]["document_info"] = dict(
                path=user_input.get(
                    "file_path",
                )
                or "document.txt",
                metadata=user_input.get("document_metadata"),
                document_type=DocumentType.INLINE_TEXT,
            )
        return run_params
