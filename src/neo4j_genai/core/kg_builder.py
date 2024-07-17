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

    def get_node_by_name(self, name: str):
        for node in self._nodes.values():
            if node.name == name:
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


class Component(Node):
    def __init__(self, name: str):
        super().__init__(name, {})
        self._graph = None

    def process(self, **kwargs: Any) -> dict:
        return {}

    def run(self, **kwargs: Any) -> dict:
        logger.info(f"Running component {self.name} with {kwargs}")
        res = self.process(**kwargs)
        self._graph.add_result_for_component(self.name, res)
        return res

    def run_all(self, data):
        input_defs = {}
        # make sure dependencies are satisfied
        # and save the inputs defs that needs to be propagated from parent components
        for prev_edge in self._graph.previous_edges(self):
            if prev_edge.start.name not in self._graph._results:
                logger.warning(f"Waiting for {prev_edge.start.name}")
                # let's wait, the run should be triggered once the last required
                # parent is done
                return {}
            prev_edge_data = prev_edge.data.get("input_defs") or {}
            input_defs.update(**prev_edge_data)
        # create component's input based on initial input data, already done component's
        # outputs and inputs definition mapping
        inputs = self._graph.get_component_inputs(self.name, input_defs, data)
        res = self.run(**inputs)
        self._graph.add_result_for_component(self.name, res)
        # run the same method for dependent (children) nodes
        for c in self._graph.next_edges(self):
            res = c.end.run_all(data)
        return res


class Pipeline(Graph):
    def __init__(self):
        super().__init__()
        self._results = {}

    def add_component(self, component):
        if component._graph:
            raise Exception("Component already part of a pipeline")
        component._graph = self
        self.add_node(component)

    def connect(self, start_component_name, end_component_name, input_defs: Optional[dict] = None):
        start_node = self.get_node_by_name(start_component_name)
        end_node = self.get_node_by_name(end_component_name)
        if start_node is None:
            raise Exception(f"Component {start_component_name} not in graph")
        if end_node is None:
            raise Exception(f"Component {end_component_name} not in graph")
        super().connect(start_node, end_node, data={"input_defs": input_defs})

    def add_result_for_component(self, name, result):
        self._results[name] = result

    def get_component_inputs(self, component_name: str, input_defs: dict, input_data: dict) -> dict:
        component_inputs = input_data.get(component_name, {})
        if input_defs:
            for input_def, mapping in input_defs.items():
                jsonpath_expr = parse(mapping)
                # input_component, param = mapping.split(".")
                # value = self._results[input_component][param]
                value = [
                    match.value
                    for match in jsonpath_expr.find(self._results)
                ]
                if value:  # TODO: how to deal with multiple matches? Is it relevant?
                    value = value[0]
                component_inputs[input_def] = value
        return component_inputs

    def run_all(self, data):
        roots = self.roots()
        r = None
        for root in roots:
            r = root.run_all(data)
        return r


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
