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
import pytest
from neo4j_graphrag.experimental.pipeline.pipeline_graph import (
    PipelineEdge,
    PipelineGraph,
    PipelineNode,
)


def test_node_alone() -> None:
    n = PipelineNode(name="node", data={})
    assert n.is_root() is True
    assert n.is_leaf() is True


def test_node_not_root() -> None:
    n = PipelineNode(name="node", data={})
    n.parents = ["other_node"]
    assert n.is_root() is False
    assert n.is_leaf() is True


def test_node_not_leaf() -> None:
    n = PipelineNode(name="node", data={})
    n.children = ["other_node"]
    assert n.is_root() is True
    assert n.is_leaf() is False


def test_graph_add_nodes() -> None:
    g: PipelineGraph[PipelineNode, PipelineEdge] = PipelineGraph()
    n1 = PipelineNode("n1", {})
    n2 = PipelineNode("n2", {})
    g.add_node(n1)
    g.add_node(n2)
    assert len(g._nodes) == 2
    edge = PipelineEdge(n1.name, n2.name, {"key": "value"})
    g.add_edge(edge)
    assert len(g._edges) == 1

    assert n1.children == [n2.name]
    assert n2.parents == [n1.name]


@pytest.fixture(scope="function")
def graph() -> PipelineGraph[PipelineNode, PipelineEdge]:
    g: PipelineGraph[PipelineNode, PipelineEdge] = PipelineGraph()
    n1 = PipelineNode("n1", {})
    n2 = PipelineNode("n2", {})
    g.add_node(n1)
    g.add_node(n2)
    edge = PipelineEdge(n1.name, n2.name, {"key": "value"})
    g.add_edge(edge)
    return g


def test_graph_roots(graph: PipelineGraph[PipelineNode, PipelineEdge]) -> None:
    roots = graph.roots()
    assert len(roots) == 1
    assert roots[0].name == "n1"


def test_graph_next_edge(graph: PipelineGraph[PipelineNode, PipelineEdge]) -> None:
    start = graph._nodes["n1"]
    next_edges = graph.next_edges(start.name)
    assert len(next_edges) == 1
    next_edge = next_edges[0]
    assert isinstance(next_edge, PipelineEdge)
    assert next_edge.start == "n1"
    assert next_edge.end == "n2"


def test_graph_prev_edge(graph: PipelineGraph[PipelineNode, PipelineEdge]) -> None:
    start = graph._nodes["n2"]
    next_edges = graph.previous_edges(start.name)
    assert len(next_edges) == 1
    next_edge = next_edges[0]
    assert isinstance(next_edge, PipelineEdge)
    assert next_edge.start == "n1"
    assert next_edge.end == "n2"


def test_graph_contains(graph: PipelineGraph[PipelineNode, PipelineEdge]) -> None:
    start = graph._nodes["n2"]
    assert start in graph


def test_graph_is_cyclic(graph: PipelineGraph[PipelineNode, PipelineEdge]) -> None:
    g: PipelineGraph[PipelineNode, PipelineEdge] = PipelineGraph()
    n1 = PipelineNode("n1", {})
    n2 = PipelineNode("n2", {})
    g.add_node(n1)
    g.add_node(n2)
    edge = PipelineEdge(n1.name, n2.name, {})
    g.add_edge(edge)
    assert g.is_cyclic() is False

    edge = PipelineEdge(n2.name, n1.name, {})
    g.add_edge(edge)
    assert g.is_cyclic() is True


def test_graph_set_node(graph: PipelineGraph[PipelineNode, PipelineEdge]) -> None:
    new_node = PipelineNode("n1", {})
    graph.set_node(new_node)
    new_node_from_graph = graph.get_node_by_name("n1")
    assert new_node_from_graph.parents == []
    assert new_node_from_graph.children == ["n2"]


def test_graph_validate_edge_bad_node_name(
    graph: PipelineGraph[PipelineNode, PipelineEdge],
) -> None:
    with pytest.raises(KeyError):
        graph.add_edge(PipelineEdge("n0", "n1", {}))
    with pytest.raises(KeyError):
        graph.add_edge(PipelineEdge("n1", "n12", {}))


def test_graph_validate_edge_no_parallel_edges(
    graph: PipelineGraph[PipelineNode, PipelineEdge],
) -> None:
    with pytest.raises(ValueError):
        graph.add_edge(PipelineEdge("n1", "n2", {}))
