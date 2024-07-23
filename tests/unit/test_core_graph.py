import pytest
from neo4j_genai.core.pipeline_graph import PipelineEdge, PipelineGraph, PipelineNode


def test_node_alone() -> None:
    n = PipelineNode(name="node", data={})
    assert n.is_root() is True
    assert n.is_leaf() is True


def test_node_not_root() -> None:
    n = PipelineNode(name="node", data={})
    n.parents = [PipelineNode(name="child", data={})]
    assert n.is_root() is False
    assert n.is_leaf() is True


def test_node_not_leaf() -> None:
    n = PipelineNode(name="node", data={})
    n.children = [PipelineNode(name="child", data={})]
    assert n.is_root() is True
    assert n.is_leaf() is False


def test_graph_add_nodes() -> None:
    g = PipelineGraph()
    n1 = PipelineNode("n1", {})
    n2 = PipelineNode("n2", {})
    g.add_node(n1)
    g.add_node(n2)
    assert len(g._nodes) == 2
    g.connect(n1, n2, {"key": "value"})
    assert len(g._edges) == 1

    assert n1.children == [n2]
    assert n2.parents == [n1]


@pytest.fixture(scope="function")
def graph() -> PipelineGraph:
    g = PipelineGraph()
    n1 = PipelineNode("n1", {})
    n2 = PipelineNode("n2", {})
    g.add_node(n1)
    g.add_node(n2)
    g.connect(n1, n2, {"key": "value"})
    return g


def test_graph_roots(graph: PipelineGraph) -> None:
    roots = graph.roots()
    assert len(roots) == 1
    assert roots[0].name == "n1"


def test_graph_next_edge(graph: PipelineGraph) -> None:
    start = graph._nodes["n1"]
    next_edges = graph.next_edges(start)
    assert len(next_edges) == 1
    next_edge = next_edges[0]
    assert isinstance(next_edge, PipelineEdge)
    assert next_edge.start.name == "n1"
    assert next_edge.end.name == "n2"


def test_graph_prev_edge(graph: PipelineGraph) -> None:
    start = graph._nodes["n2"]
    next_edges = graph.previous_edges(start)
    assert len(next_edges) == 1
    next_edge = next_edges[0]
    assert isinstance(next_edge, PipelineEdge)
    assert next_edge.start.name == "n1"
    assert next_edge.end.name == "n2"


def test_graph_contains(graph: PipelineGraph) -> None:
    start = graph._nodes["n2"]
    assert start in graph
