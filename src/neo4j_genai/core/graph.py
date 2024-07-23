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
"""
Basic graph structure for Pipeline
"""

from __future__ import annotations

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

    def set_node(self, node: Node) -> None:
        if node not in self:
            raise ValueError(f"Node {node.name} does not exist")
        self._nodes[node.name] = node
        for edge in self._edges:
            if edge.start.name == node.name:
                edge.start = node
            if edge.end.name == node.name:
                edge.end = node

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

    def __contains__(self, node: Node | str) -> bool:
        if isinstance(node, Node):
            return node.name in self._nodes
        return node in self._nodes

    def dfs(self, visited: set[Node], node: Node) -> bool:
        if node in visited:
            return True
        else:
            for edge in self.next_edges(node):
                neighbour = edge.end
                if self.dfs(visited | {node}, neighbour):
                    return True
            return False

    def is_cyclic(self) -> bool:
        for node in self._nodes.values():
            if self.dfs(set(), node):
                return True
        return False
