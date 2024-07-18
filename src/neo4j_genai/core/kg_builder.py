"""
More complex Pipeline implementation, dealing with Branching and Aggregation:

Branching:
      |
    /   \
e.g. DocumentChunker => ERExtractor, Embedder

Aggregation:
    \  /
     |
e.g. SchemaBuilder + Chunker => ERExtractor

"""
import abc
import logging
from typing import Any, Optional

from jsonpath_ng import parse

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Node:
    def __init__(self, name: str, data: dict) -> None:
        self.name = name
        self.data = data
        self.parents = []
        self.children = []

    def is_root(self):
        return len(self.parents) == 0

    def is_leaf(self):
        return len(self.children) == 0


class Edge:
    def __init__(self, start: Node, end: Node, data: Optional[dict] = None):
        self.start = start
        self.end = end
        self.data = data


class Graph:
    def __init__(self):
        self._nodes = {}
        self._edges = []

    def add_node(self, node: Node):
        self._nodes[node.name] = node

    def connect(self, start: Node, end: Node, data: dict):
        self._edges.append(Edge(start, end, data))
        self._nodes[end.name].parents.append(start)
        self._nodes[start.name].children.append(end)

    def get_node_by_name(self, name: str, raise_exception: bool = False) -> Node:
        node = self._nodes.get(name)
        if node is None and raise_exception:
            raise KeyError(f"Component {name} not in graph")
        return node

    def roots(self):
        root = []
        for node in self._nodes.values():
            if node.is_root():
                root.append(node)
        return root

    def next_edges(self, node):
        res = []
        for edge in self._edges:
            if edge.start == node:
                res.append(edge)
        return res

    def previous_edges(self, node):
        res = []
        for edge in self._edges:
            if edge.end == node:
                res.append(edge)
        return res

    def __contains__(self, node: Node):
        return node.name in self._nodes


class MissingDependencyError(Exception):
    pass


class Component(Node):
    def __init__(self, name: str):
        super().__init__(name, {})
        self._graph = None

    def process(self, **kwargs: Any) -> dict:
        return {}

    def execute(self, **kwargs: Any) -> dict:
        logger.info(f"Running component {self.name} with {kwargs}")
        res = self.process(**kwargs)
        return res

    def get_input_defs_from_parents(self):
        input_defs = {}
        # make sure dependencies are satisfied
        # and save the inputs defs that needs to be propagated from parent components
        for prev_edge in self._graph.previous_edges(self):
            if self._graph.get_results_for_component(prev_edge.start.name) is None:
                logger.warning(f"Waiting for {prev_edge.start.name}")
                # let's wait, the run should be triggered once the last required
                # parent is done
                raise MissingDependencyError(f"{prev_edge.start.name} not ready")
            prev_edge_data = prev_edge.data.get("input_defs") or {}
            input_defs.update(**prev_edge_data)
        return input_defs

    def run_all(self, data):
        # make sure dependencies are satisfied
        # and save the inputs defs that needs to be
        # propagated from parent components
        try:
            input_defs = self.get_input_defs_from_parents()
        except MissingDependencyError:
            return {"status": "WAITING"}
        # create component's input based on initial input data,
        # already done component's outputs and inputs definition mapping
        inputs = self._graph.get_component_inputs(self.name, input_defs, data)
        # execute component task
        res = self.execute(**inputs)
        # save its results
        self._graph.add_result_for_component(self.name, res, is_final=self.is_leaf())
        # trigger execution of children nodes
        for c in self._graph.next_edges(self):
            res = c.end.run_all(data)
        return res


class Store(abc.ABC):
    """An interface to save component outputs"""

    @abc.abstractmethod
    def add(self, key: str, value: Any, overwrite: bool = True) -> None:
        """
        Args:
            key (str): The key to access the data.
        """
        pass

    @abc.abstractmethod
    def get(self, key: str) -> Any:
        pass

    @abc.abstractmethod
    def find_all(self, pattern) -> list:
        pass

    def all(self):
        raise NotImplementedError()


class InMemoryStore(Store):
    """Simple in-memory store.
    Saves each component's results in a _data dict."""

    def __init__(self):
        self._data = {}

    def add(self, key: str, value: Any, overwrite: bool = True) -> None:
        if (not overwrite) and key in self._data:
            raise KeyError(f"{key} already exists")
        self._data[key] = value

    def get(self, key: str) -> Any:
        return self._data.get(key)

    def find_all(self, pattern) -> list:
        jsonpath_expr = parse(pattern)
        # input_component, param = mapping.split(".")
        # value = self._results[input_component][param]
        value = [
            match.value
            for match in jsonpath_expr.find(self._data)
        ]
        return value

    def all(self):
        return self._data


class Pipeline(Graph):
    def __init__(self, store: Optional[Store] = None):
        super().__init__()
        self._store = store or InMemoryStore()
        self._final_results = InMemoryStore()

    def add_component(self, component):
        if component._graph:
            raise Exception("Component already part of a pipeline")
        component._graph = self
        self.add_node(component)

    def connect(self, start_component_name, end_component_name,
                input_defs: Optional[dict[str, str]] = None):
        start_node = self.get_node_by_name(start_component_name, raise_exception=True)
        end_node = self.get_node_by_name(end_component_name, raise_exception=True)
        super().connect(start_node, end_node, data={"input_defs": input_defs})

    def add_result_for_component(self, name, result, is_final: bool = False):
        self._store.add(name, result)
        if is_final:
            self._final_results.add(name, result)

    def get_results_for_component(self, name: str):
        return self._store.get(name)

    def get_component_inputs(self, component_name: str, input_defs: dict,
                             input_data: dict) -> dict:
        component_inputs = input_data.get(component_name, {})
        if input_defs:
            for input_def, mapping in input_defs.items():
                value = self._store.find_all(mapping)
                if value:  # TODO: how to deal with multiple matches? Is it relevant?
                    value = value[0]
                component_inputs[input_def] = value
        return component_inputs

    def run_all(self, data):
        roots = self.roots()
        for root in roots:
            root.run_all(data)
        return self._final_results.all()


class DocumentChunker(Component):
    def process(self, text: str):
        return {
            "chunks": [t.strip() for t in text.split(".") if t.strip()]
        }


class SchemaBuilder(Component):
    def process(self, schema: dict):
        return {"schema": schema}


class ERExtractor(Component):
    def process(self, chunks: list[str], schema: str) -> dict:
        return {
            "data": {
                "entities": [{"label": "Person", "properties": {"name": "John Doe"}}],
                "relations": []
            }
        }


class Writer(Component):
    def process(self, entities: dict, relations: dict) -> dict:
        return {
            "status": "OK",
            "entities": entities,
            "relations": relations,
        }


if __name__ == '__main__':
    pipe = Pipeline()
    pipe.add_component(DocumentChunker("chunker"))
    pipe.add_component(SchemaBuilder("schema"))
    pipe.add_component(ERExtractor("extractor"))
    pipe.add_component(Writer("writer"))
    pipe.connect("chunker", "extractor", input_defs={
        "chunks": "chunker.chunks"
    })
    pipe.connect("schema", "extractor", input_defs={
        "schema": "schema.schema"
    })
    pipe.connect("extractor", "writer", input_defs={
        "entities": "extractor.data.entities",
        "relations": "extractor.data.relations",
    })

    pipe_inputs = {
        "chunker":
            {
                "text": "Graphs are everywhere. "
                        "GraphRAG is the future of Artificial Intelligence. "
                        "Robots are already running the world."
            },
        "schema": {
            "schema": "Person OWNS House"
        }
    }
    print(pipe.run_all(pipe_inputs))
    # print(asyncio.run(pipe.arun(pipe_inputs)))
