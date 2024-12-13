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

from typing import Any, List, Optional, Sequence, Union

import neo4j
from pydantic import ValidationError

from neo4j_graphrag.embeddings import Embedder
from neo4j_graphrag.experimental.components.types import LexicalGraphConfig
from neo4j_graphrag.experimental.pipeline.config.runner import PipelineRunner
from neo4j_graphrag.experimental.pipeline.config.template_pipeline import (
    SimpleKGPipelineConfig,
)
from neo4j_graphrag.experimental.pipeline.exceptions import PipelineDefinitionError
from neo4j_graphrag.experimental.pipeline.pipeline import PipelineResult
from neo4j_graphrag.experimental.pipeline.types import (
    EntityInputType,
    RelationInputType,
)
from neo4j_graphrag.generation.prompts import ERExtractionTemplate
from neo4j_graphrag.llm.base import LLMInterface


class SimpleKGPipeline:
    """
    A class to simplify the process of building a knowledge graph from text documents.
    It abstracts away the complexity of setting up the pipeline and its components.

    Args:
        llm (LLMInterface): An instance of an LLM to use for entity and relation extraction.
        driver (neo4j.Driver): A Neo4j driver instance for database connection.
        embedder (Embedder): An instance of an embedder used to generate chunk embeddings from text chunks.
        entities (Optional[List[Union[str, dict[str, str], SchemaEntity]]]): A list of either:

            - str: entity labels
            - dict: following the SchemaEntity schema, ie with label, description and properties keys

        relations (Optional[List[Union[str, dict[str, str], SchemaRelation]]]): A list of either:

            - str: relation label
            - dict: following the SchemaRelation schema, ie with label, description and properties keys

        potential_schema (Optional[List[tuple]]): A list of potential schema relationships.
        from_pdf (bool): Determines whether to include the PdfLoader in the pipeline.
                         If True, expects `file_path` input in `run` methods.
                         If False, expects `text` input in `run` methods.
        text_splitter (Optional[Any]): A text splitter component. Defaults to FixedSizeSplitter().
        pdf_loader (Optional[Any]): A PDF loader component. Defaults to PdfLoader().
        kg_writer (Optional[Any]): A knowledge graph writer component. Defaults to Neo4jWriter().
        on_error (str): Error handling strategy for the Entity and relation extractor. Defaults to "IGNORE", where chunk will be ignored if extraction fails. Possible values: "RAISE" or "IGNORE".
        perform_entity_resolution (bool): Merge entities with same label and name. Default: True
        prompt_template (str): A custom prompt template to use for extraction.
        lexical_graph_config (Optional[LexicalGraphConfig], optional): Lexical graph configuration to customize node labels and relationship types in the lexical graph.
    """

    def __init__(
        self,
        llm: LLMInterface,
        driver: neo4j.Driver,
        embedder: Embedder,
        entities: Optional[Sequence[EntityInputType]] = None,
        relations: Optional[Sequence[RelationInputType]] = None,
        potential_schema: Optional[List[tuple[str, str, str]]] = None,
        from_pdf: bool = True,
        text_splitter: Optional[Any] = None,
        pdf_loader: Optional[Any] = None,
        kg_writer: Optional[Any] = None,
        on_error: str = "IGNORE",
        prompt_template: Union[ERExtractionTemplate, str] = ERExtractionTemplate(),
        perform_entity_resolution: bool = True,
        lexical_graph_config: Optional[LexicalGraphConfig] = None,
        neo4j_database: Optional[str] = None,
    ):
        try:
            config = SimpleKGPipelineConfig(
                # argument type are fixed in the Config object
                llm_config=llm,  # type: ignore[arg-type]
                neo4j_config=driver,  # type: ignore[arg-type]
                embedder_config=embedder,  # type: ignore[arg-type]
                entities=entities or [],
                relations=relations or [],
                potential_schema=potential_schema,
                from_pdf=from_pdf,
                pdf_loader=pdf_loader,
                kg_writer=kg_writer,
                text_splitter=text_splitter,
                on_error=on_error,  # type: ignore[arg-type]
                prompt_template=prompt_template,
                perform_entity_resolution=perform_entity_resolution,
                lexical_graph_config=lexical_graph_config,
                neo4j_database=neo4j_database,
            )
        except ValidationError as e:
            raise PipelineDefinitionError() from e

        self.runner = PipelineRunner.from_config(config)

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
        return await self.runner.run({"file_path": file_path, "text": text})
