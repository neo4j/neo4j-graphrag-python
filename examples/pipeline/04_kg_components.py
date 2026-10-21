"""Compose existing knowledge-graph components in a dataflow pipeline.

Text goes through three independent, swappable stages:

    text -> FixedSizeSplitter -> entity extractor -> graph sink

* The splitter is the real `FixedSizeSplitter`, used through its synchronous
  `iter_chunks` method with `flat_map` (one text -> many chunks).
* Components are not callable and their `run` methods are `async`, so they
  cannot be given to `map` as they are (you would get unawaited coroutines).
  Use a synchronous helper with `map`/`flat_map`, or `map_async_chunked` for
  `run` (see the second pipeline below).
* The extractor is a small stand-in so the example runs offline. To use an LLM,
  swap in `LLMEntityRelationExtractor` (see the commented snippet below): the
  rest of the pipeline does not change.
* The sink prints the graph by default. Set NEO4J_URI, NEO4J_USERNAME and
  NEO4J_PASSWORD to write it to Neo4j with `Neo4jWriter` instead.

    python examples/pipeline/04_kg_components.py
"""

import asyncio
import os

import neo4j
from neo4j_graphrag.components.kg_writer import KGWriter, Neo4jWriter
from neo4j_graphrag.components.text_splitters.fixed_size_splitter import (
    FixedSizeSplitter,
)
from neo4j_graphrag.components.types import Neo4jGraph, Neo4jNode, TextChunk
from neo4j_graphrag.pipeline import Pipeline, Sink

TEXT = (
    "Alice works at Acme. Bob works at Initech. Carol lives in Paris. "
    "Dave works at Acme and knows Alice."
)


# Stage 1: splitting. Configured through the component's own constructor.
splitter = FixedSizeSplitter(chunk_size=40, chunk_overlap=0, approximate=True)


# Stage 2: extraction. Any `TextChunk -> Neo4jGraph` function (or async function)
# fits here. This one finds capitalised words and makes a node out of each.
def extract_entities(chunk: TextChunk) -> Neo4jGraph:
    names = {w.strip(".,") for w in chunk.text.split() if w[:1].isupper()}
    return Neo4jGraph(
        nodes=[Neo4jNode(id=n, label="Entity", properties={"name": n}) for n in names]
    )


# With an LLM, use the real component instead (needs an API key). Its async `run`
# takes the chunks produced by `splitter.run` (a `TextChunks`) and any extra
# argument can be bound with `functools.partial`:
#
#     from functools import partial
#
#     from neo4j_graphrag.components.entity_relation_extractor import (
#         LLMEntityRelationExtractor,
#     )
#     from neo4j_graphrag.llm import OpenAILLM
#
#     extractor = LLMEntityRelationExtractor(llm=OpenAILLM(model_name="gpt-4o"))
#     (
#         Pipeline([TEXT])
#         .map_async_chunked(splitter.run)  # str -> TextChunks
#         .map_async_chunked(partial(extractor.run, schema=my_schema))  # -> Neo4jGraph
#         ...
#     )


def merge_graphs(graphs: list[Neo4jGraph]) -> Neo4jGraph:
    nodes = {n.id: n for g in graphs for n in g.nodes}  # de-duplicate on id
    return Neo4jGraph(nodes=list(nodes.values()))


# Stage 3: writing. A sink can wrap any `KGWriter` component.
class PrintSink(Sink[Neo4jGraph]):
    def write(self, element: Neo4jGraph) -> None:
        print(f"graph with nodes: {sorted(n.id for n in element.nodes)}")


class KGWriterSink(Sink[Neo4jGraph]):
    def __init__(self, writer: KGWriter) -> None:
        self.writer = writer

    def write(self, element: Neo4jGraph) -> None:
        result = asyncio.run(self.writer.run(element))
        print(f"write status: {result.status}")


def build_sink() -> Sink[Neo4jGraph]:
    if uri := os.getenv("NEO4J_URI"):
        auth = (os.getenv("NEO4J_USERNAME", "neo4j"), os.environ["NEO4J_PASSWORD"])
        driver = neo4j.GraphDatabase.driver(uri, auth=auth)
        return KGWriterSink(Neo4jWriter(driver=driver, clean_db=False))
    return PrintSink()


(
    Pipeline([TEXT])
    .flat_map(splitter.iter_chunks, label="split")
    .map(extract_entities, label="extract")
    .grouped(1000, label="batch")  # one batch -> one graph
    .map(merge_graphs, label="merge")
    .to_sink(build_sink(), label="write")
)

# The same first stage, using the component's async `run` method instead.
# `run` returns a `TextChunks` model, so unwrap it with `flat_map`. Components
# whose `run` takes more than one argument (like the extractor) need a
# `partial`, as in the commented snippet above.
#
# NB: async stages are evaluated with `asyncio.run`, so they cannot be used from
# inside a running event loop (a notebook, or an `async def main()`).
chunks = (
    Pipeline([TEXT])
    .map_async_chunked(splitter.run, map_batch_size=10, label="split")
    .flat_map(lambda text_chunks: text_chunks.chunks, label="unwrap")
    .collect()
)
print(f"{len(chunks)} chunks via the async `run` method")
