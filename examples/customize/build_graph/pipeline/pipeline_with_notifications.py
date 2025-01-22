"""This example demonstrates how to use event callback to receive notifications
about the pipeline progress.
"""

from __future__ import annotations

import asyncio
import logging

import neo4j
from neo4j_graphrag.experimental.components.kg_writer import Neo4jWriter
from neo4j_graphrag.experimental.components.lexical_graph import LexicalGraphBuilder
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import (
    FixedSizeSplitter,
)
from neo4j_graphrag.experimental.pipeline import Pipeline
from neo4j_graphrag.experimental.pipeline.pipeline import PipelineResult
from neo4j_graphrag.experimental.pipeline.types import Event

logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.WARNING)


async def event_handler(event: Event) -> None:
    """Function can do anything about the event,
    here we're just logging it if it's a pipeline-level event.
    """
    if event.event_type.is_pipeline_event:
        logger.warning(event)


async def main(neo4j_driver: neo4j.Driver) -> PipelineResult:
    """This is where we define and run the Lexical Graph builder pipeline, instantiating
    a few components:

    - Text Splitter: to split the text into manageable chunks of fixed size
    - Chunk Embedder: to embed the chunks' text
    - Lexical Graph Builder: to build the lexical graph, ie creating the chunk nodes and relationships between them
    - KG writer: save the lexical graph to Neo4j
    """
    pipe = Pipeline(
        callback=event_handler,
    )
    # define the components
    pipe.add_component(
        FixedSizeSplitter(chunk_size=300, chunk_overlap=10, approximate=False),
        "splitter",
    )
    pipe.add_component(
        LexicalGraphBuilder(),
        "lexical_graph_builder",
    )
    pipe.add_component(Neo4jWriter(neo4j_driver), "writer")
    # define the execution order of component
    # and how the output of previous components must be used
    pipe.connect(
        "splitter", "lexical_graph_builder", input_config={"text_chunks": "splitter"}
    )
    pipe.connect(
        "lexical_graph_builder",
        "writer",
        input_config={
            "graph": "lexical_graph_builder.graph",
            "lexical_graph_config": "lexical_graph_builder.config",
        },
    )
    # user input:
    # the initial text
    # and the list of entities and relations we are looking for
    pipe_inputs = {
        "splitter": {
            "text": """Albert Einstein was a German physicist born in 1879 who
            wrote many groundbreaking papers especially about general relativity
            and quantum mechanics. He worked for many different institutions, including
            the University of Bern in Switzerland and the University of Oxford."""
        },
        "lexical_graph_builder": {
            "document_info": {
                # 'path' can be anything
                "path": "example/pipeline_with_notifications"
            },
        },
    }
    # run the pipeline
    return await pipe.run(pipe_inputs)


if __name__ == "__main__":
    with neo4j.GraphDatabase.driver(
        "bolt://localhost:7687", auth=("neo4j", "password")
    ) as driver:
        print(asyncio.run(main(driver)))
