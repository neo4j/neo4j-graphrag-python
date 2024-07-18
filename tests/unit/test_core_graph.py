import pytest

from neo4j_genai.core.graph import Node, Edge, Graph


def test_node_alone():
    n = Node(name="node", data={})
    assert n.is_root() is True
    assert n.is_leaf() is True


def test_node_not_root():
    n = Node(name="node", data={})
    n.parents = [Node(name="child", data={})]
    assert n.is_root() is False
    assert n.is_leaf() is True


def test_node_not_leaf():
    n = Node(name="node", data={})
    n.children = [Node(name="child", data={})]
    assert n.is_root() is True
    assert n.is_leaf() is False


def test_graph_add_nodes():
    g = Graph()
    n1 = Node("n1", {})
    n2 = Node("n2", {})
    g.add_node(n1)
    g.add_node(n2)
    assert len(g._nodes) == 2
    g.connect(n1, n2, {"key": "value"})
    assert len(g._edges) == 1

    assert n1.children == [n2]
    assert n2.parents == [n1]


@pytest.fixture
def graph():
    g = Graph()
    n1 = Node("n1", {})
    n2 = Node("n2", {})
    g.add_node(n1)
    g.add_node(n2)
    g.connect(n1, n2, {"key": "value"})
    return g


def test_graph_roots(graph):
    roots = graph.roots()
    assert len(roots) == 1
    assert roots[0].name == "n1"


def test_graph_next_edge(graph):
    start = graph._nodes["n1"]
    next_edges = graph.next_edges(start)
    assert len(next_edges) == 1
    next_edge = next_edges[0]
    assert isinstance(next_edge, Edge)
    assert next_edge.start.name == "n1"
    assert next_edge.end.name == "n2"


def test_graph_prev_edge(graph):
    start = graph._nodes["n2"]
    next_edges = graph.previous_edges(start)
    assert len(next_edges) == 1
    next_edge = next_edges[0]
    assert isinstance(next_edge, Edge)
    assert next_edge.start.name == "n1"
    assert next_edge.end.name == "n2"


def test_graph_contains(graph):
    start = graph._nodes["n2"]
    assert start in graph
