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
"""Smoke tests for the dataflow pipeline examples in ``examples/pipeline``.

The examples are plain scripts that run at import time, so they are executed
with ``runpy`` and their output is checked.
"""

from __future__ import annotations

import runpy
from functools import partial
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from neo4j_graphrag.components.entity_relation_extractor import (
    LLMEntityRelationExtractor,
)
from neo4j_graphrag.components.kg_writer import KGWriterModel
from neo4j_graphrag.components.schema import GraphSchema
from neo4j_graphrag.components.types import Neo4jGraph, Neo4jNode
from neo4j_graphrag.llm import LLMInterface
from neo4j_graphrag.llm.types import LLMResponse
from neo4j_graphrag.pipeline import Pipeline

EXAMPLES_DIR = Path(__file__).parents[3] / "examples" / "pipeline"
ALL_NODES = "['Acme', 'Alice', 'Bob', 'Carol', 'Dave', 'Initech', 'Paris']"


def run_example(name: str) -> dict[str, Any]:
    return runpy.run_path(str(EXAMPLES_DIR / name), run_name="example")


def test_basic_pipeline(capsys: pytest.CaptureFixture[str]) -> None:
    run_example("01_basic_pipeline.py")
    out = capsys.readouterr().out
    expected = ["alice works at acme", "bob works at initech", "carol lives in paris"]
    assert str(expected) in out
    for sentence in expected:
        assert f"-> {sentence}" in out  # written to the sink
    assert "['ALICE WORKS AT ACME', 'BOB WORKS AT INITECH']" in out


def test_configuring_stages(capsys: pytest.CaptureFixture[str]) -> None:
    run_example("02_configuring_stages.py")
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("['> chunk number 1…'")
    assert lines[1] == "['# chunk…', '# chunk…', '# chunk…']"
    assert lines[2] == "[4, 4, 2]"  # grouped(4) over 10 elements
    assert lines[3] == "[290]"  # reduce


def test_error_handling_and_observers(capsys: pytest.CaptureFixture[str]) -> None:
    run_example("03_error_handling_and_observers.py")
    out = capsys.readouterr().out
    assert "aborted: invalid literal for int()" in out
    assert "Ok(value=1998)" in out
    assert "Err(exception=ValueError(" in out
    assert "[1998, 2004, 2021]" in out  # filter_ok drops the failure
    assert "[1999, 2005, 2022] (1 failure)" in out  # on_error counted it


class TestKGComponentsExample:
    @pytest.fixture(autouse=True)
    def no_neo4j(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("NEO4J_URI", raising=False)

    def test_default_run_prints_merged_graph(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        run_example("04_kg_components.py")
        out = capsys.readouterr().out
        assert f"graph with nodes: {ALL_NODES}" in out
        assert "3 chunks via the async `run` method" in out

    def test_writes_to_neo4j_when_uri_is_set(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        driver_factory = MagicMock()
        monkeypatch.setattr("neo4j.GraphDatabase.driver", driver_factory)
        monkeypatch.setenv("NEO4J_URI", "neo4j://example:7687")
        written: list[Neo4jGraph] = []

        class FakeNeo4jWriter:
            def __init__(self, driver: Any, clean_db: bool) -> None:
                assert clean_db is False

            async def run(self, graph: Neo4jGraph) -> KGWriterModel:
                written.append(graph)
                return KGWriterModel(status="SUCCESS")

        monkeypatch.setattr(
            "neo4j_graphrag.components.kg_writer.Neo4jWriter", FakeNeo4jWriter
        )

        run_example("04_kg_components.py")

        assert "write status: SUCCESS" in capsys.readouterr().out
        assert len(written) == 1
        assert sorted(n.id for n in written[0].nodes) == [
            "Acme",
            "Alice",
            "Bob",
            "Carol",
            "Dave",
            "Initech",
            "Paris",
        ]
        # the driver is used as a context manager, so it is closed
        driver_factory.return_value.__exit__.assert_called_once()

    def test_sink_raises_when_writer_reports_failure(self) -> None:
        example = run_example("04_kg_components.py")

        class FailingWriter:
            async def run(self, graph: Neo4jGraph) -> KGWriterModel:
                return KGWriterModel(status="FAILURE", metadata={"error": "boom"})

        sink = example["KGWriterSink"](FailingWriter())
        graph = Neo4jGraph(nodes=[Neo4jNode(id="a", label="Entity")])
        with pytest.raises(RuntimeError, match="boom"):
            sink.write(graph)

    def test_extract_entities_and_merge(self) -> None:
        example = run_example("04_kg_components.py")
        chunk = example["splitter"].iter_chunks("Alice knows Bob.")
        graphs = [example["extract_entities"](c) for c in chunk]
        merged = example["merge_graphs"](graphs + graphs)  # duplicates collapse
        assert sorted(n.id for n in merged.nodes) == ["Alice", "Bob"]

    def test_real_extractor_with_partial_as_documented_in_the_example(self) -> None:
        """The commented LLM snippet in the example: splitter.run, then
        partial(extractor.run, schema=...), both through map_async_chunked."""
        example = run_example("04_kg_components.py")
        llm = MagicMock(spec=LLMInterface)
        llm.ainvoke.return_value = LLMResponse(
            content='{"nodes": [{"id": "0", "label": "Person", '
            '"properties": {"name": "Alice"}}], "relationships": []}'
        )
        extractor = LLMEntityRelationExtractor(llm=llm)

        graphs = (
            Pipeline(["Alice works at Acme. Bob lives in Paris."])
            .map_async_chunked(example["splitter"].run)
            .map_async_chunked(
                partial(extractor.run, schema=GraphSchema(node_types=()))
            )
            .collect()
        )

        assert len(graphs) == 1
        assert isinstance(graphs[0], Neo4jGraph)
        assert any(n.label == "Person" for n in graphs[0].nodes)
        assert llm.ainvoke.await_count >= 1  # the LLM was called per chunk
