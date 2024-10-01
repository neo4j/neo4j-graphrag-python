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
import neo4j
import pytest
from unittest.mock import MagicMock

from neo4j_graphrag.experimental.components.schema import SchemaEntity, SchemaRelation
from neo4j_graphrag.experimental.pipeline.kg_builder import KnowledgeGraphBuilder
from neo4j_graphrag.llm import LLMResponse
from neo4j_graphrag.llm.base import LLMInterface


def test_knowledge_graph_builder_init_with_text():
    llm = MagicMock(spec=LLMInterface)
    llm.ainvoke.return_value = LLMResponse(
        content='{"nodes": [{"id": "0", "label": "Person", "properties": {}}], "relationships": []}'
    )
    driver = MagicMock(spec=neo4j.Driver)

    text_input = "May thy knife chip and shatter."
    kg_builder = KnowledgeGraphBuilder(
        llm=llm,
        driver=driver,
        text=text_input,
    )

    assert kg_builder.llm == llm
    assert kg_builder.driver == driver
    assert kg_builder.text == text_input
    assert kg_builder.file_path is None
    assert kg_builder.entities == []
    assert kg_builder.relations == []
    assert kg_builder.potential_schema == []


def test_knowledge_graph_builder_init_with_file_path():
    llm = MagicMock(spec=LLMInterface)
    llm.ainvoke.return_value = LLMResponse(
        content='{"nodes": [{"id": "0", "label": "Person", "properties": {}}], "relationships": []}'
    )
    driver = MagicMock(spec=neo4j.Driver)

    file_path = "path/to/test.pdf"
    kg_builder = KnowledgeGraphBuilder(
        llm=llm,
        driver=driver,
        file_path=file_path,
    )

    assert kg_builder.llm == llm
    assert kg_builder.driver == driver
    assert kg_builder.text is None
    assert kg_builder.file_path == file_path
    assert kg_builder.entities == []
    assert kg_builder.relations == []
    assert kg_builder.potential_schema == []


def test_knowledge_graph_builder_init_with_both_inputs():
    llm = MagicMock(spec=LLMInterface)
    llm.ainvoke.return_value = LLMResponse(
        content='{"nodes": [{"id": "0", "label": "Person", "properties": {}}], "relationships": []}'
    )
    driver = MagicMock(spec=neo4j.Driver)

    text_input = "May thy knife chip and shatter."
    file_path = "path/to/test.pdf"

    with pytest.raises(ValueError) as exc_info:
        KnowledgeGraphBuilder(
            llm=llm,
            driver=driver,
            text=text_input,
            file_path=file_path,
        )

    assert "Exactly one of 'file_path' or 'text' must be provided." in str(
        exc_info.value
    )


def test_knowledge_graph_builder_init_with_no_inputs():
    llm = MagicMock(spec=LLMInterface)
    llm.ainvoke.return_value = LLMResponse(
        content='{"nodes": [{"id": "0", "label": "Person", "properties": {}}], "relationships": []}'
    )
    driver = MagicMock(spec=neo4j.Driver)

    with pytest.raises(ValueError) as exc_info:
        KnowledgeGraphBuilder(
            llm=llm,
            driver=driver,
        )

    assert "Exactly one of 'file_path' or 'text' must be provided." in str(
        exc_info.value
    )


def test_knowledge_graph_builder_document_info_with_file():
    llm = MagicMock(spec=LLMInterface)
    llm.ainvoke.return_value = LLMResponse(
        content='{"nodes": [{"id": "0", "label": "Person", "properties": {}}], "relationships": []}'
    )
    driver = MagicMock(spec=neo4j.Driver)

    file_path = "path/to/test.pdf"
    kg_builder = KnowledgeGraphBuilder(
        llm=llm,
        driver=driver,
        file_path=file_path,
    )

    assert "loader" in kg_builder.pipe_inputs
    assert kg_builder.pipe_inputs["loader"] == {"filepath": file_path}
    assert "extractor" not in kg_builder.pipe_inputs


def test_knowledge_graph_builder_document_info_with_text():
    llm = MagicMock(spec=LLMInterface)
    llm.ainvoke.return_value = LLMResponse(
        content='{"nodes": [{"id": "0", "label": "Person", "properties": {}}], "relationships": []}'
    )
    driver = MagicMock(spec=neo4j.Driver)

    text_input = "May thy knife chip and shatter."
    kg_builder = KnowledgeGraphBuilder(
        llm=llm,
        driver=driver,
        text=text_input,
    )

    assert "splitter" in kg_builder.pipe_inputs
    assert kg_builder.pipe_inputs["splitter"] == {"text": text_input}
    assert "extractor" in kg_builder.pipe_inputs
    assert kg_builder.pipe_inputs["extractor"] == {
        "document_info": {"path": "direct_text_input"}
    }


def test_knowledge_graph_builder_with_entities_and_file():
    llm = MagicMock(spec=LLMInterface)
    llm.ainvoke.return_value = LLMResponse(
        content='{"nodes": [{"id": "0", "label": "Person", "properties": {}}], "relationships": []}'
    )
    driver = MagicMock(spec=neo4j.Driver)

    file_path = "path/to/test.pdf"
    entities = [SchemaEntity(label="Document")]
    relations = [SchemaRelation(label="CONTAINS")]
    potential_schema = [("Document", "CONTAINS", "Section")]

    kg_builder = KnowledgeGraphBuilder(
        llm=llm,
        driver=driver,
        file_path=file_path,
        entities=entities,
        relations=relations,
        potential_schema=potential_schema,
    )

    assert kg_builder.entities == entities
    assert kg_builder.relations == relations
    assert kg_builder.potential_schema == potential_schema
