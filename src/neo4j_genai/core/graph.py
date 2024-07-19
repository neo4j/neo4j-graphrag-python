"""
Basic graph structure for Pipeline
"""

from typing import Any, Optional


class Node:
    def __init__(self, name: str, data: dict[str, Any]) -> None:
        self.name = name
        self.data = data
        self.parents: list[Node] = []
        self.children: list[Node] = []

    def is_root(self) -> bool:
        return len(self.parents) == 0

    def is_leaf(self) -> bool:
        return len(self.children) == 0


class Edge:
    def __init__(self, start: Node, end: Node, data: Optional[dict[str, Any]] = None):
        self.start = start
        self.end = end
        self.data = data


class Graph:
    def __init__(self) -> None:
        self._nodes: dict[str, Node] = {}
        self._edges: list[Edge] = []

    def add_node(self, node: Node) -> None:
        if node in self:
            raise ValueError(f"Node {node.name} already exists")
        self._nodes[node.name] = node

    def connect(self, start: Node, end: Node, data: dict[str, Any]) -> None:
        self._edges.append(Edge(start, end, data))
        self._nodes[end.name].parents.append(start)
        self._nodes[start.name].children.append(end)

    def get_node_by_name(self, name: str, raise_exception: bool = False) -> Node:
        node = self._nodes.get(name)
        if node is None and raise_exception:
            raise KeyError(f"Component {name} not in graph")
        return node  # type: ignore

    def roots(self) -> list[Node]:
        root = []
        for node in self._nodes.values():
            if node.is_root():
                root.append(node)
        return root

    def next_edges(self, node: Node) -> list[Edge]:
        res = []
        for edge in self._edges:
            if edge.start == node:
                res.append(edge)
        return res

    def previous_edges(self, node: Node) -> list[Edge]:
        res = []
        for edge in self._edges:
            if edge.end == node:
                res.append(edge)
        return res

    def __contains__(self, node: Node) -> bool:
        return node.name in self._nodes

    def _check_stack(self, v, visited, stack):
        visited[v.name] = True
        stack[v.name] = True
        for linked_vertices in self.next_edges(v):
            next_node = linked_vertices.end
            if not visited[next_node.name]:
                if self._check_stack(next_node, visited, stack):
                    return True
            elif stack[next_node.name]:
                return True
        stack[v.name] = False
        return False

    def __is_cyclic(self) -> bool:
        visited = {node: False for node in self._nodes}
        stack = {node: False for node in self._nodes}
        for node_name, node in self._nodes.items():
            if not visited[node_name]:
                if self._check_stack(node, visited, stack):
                    return True
        return False

    def dfs(self, visited, node):
        if node in visited:
            return True
        else:
            for edge in self.next_edges(node):
                neighbour = edge.end
                if self.dfs(visited | {node}, neighbour):
                    return True
            return False

    def is_cyclic(self):
        for node in self._nodes.values():
            if self.dfs(set(), node):
                return True
        return False
