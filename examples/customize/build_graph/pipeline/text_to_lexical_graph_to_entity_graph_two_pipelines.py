"""In this example, we implement two pipelines:

1. A first pipeline reads a text, chunks it and save the chunks into Neo4j (the lexical graph)
2. A second pipeline reads the chunks from the database, performs entity and relation extraction and save the extracted entities into the database.
"""

from __future__ import annotations

import asyncio

from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.experimental.components.embedder import TextChunkEmbedder
from neo4j_graphrag.experimental.components.entity_relation_extractor import (
    LLMEntityRelationExtractor,
)
from neo4j_graphrag.experimental.components.kg_writer import Neo4jWriter
from neo4j_graphrag.experimental.components.lexical_graph import LexicalGraphBuilder
from neo4j_graphrag.experimental.components.neo4j_reader import Neo4jChunkReader
from neo4j_graphrag.experimental.components.schema import (
    SchemaBuilder,
    NodeType,
    PropertyType,
    RelationshipType,
)
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import (
    FixedSizeSplitter,
)
from neo4j_graphrag.experimental.components.types import LexicalGraphConfig
from neo4j_graphrag.experimental.pipeline import Pipeline
from neo4j_graphrag.experimental.pipeline.pipeline import PipelineResult
from neo4j_graphrag.llm import LLMInterface, OpenAILLM

import neo4j


async def build_lexical_graph(
    neo4j_driver: neo4j.Driver,
    lexical_graph_config: LexicalGraphConfig,
    text: str,
) -> PipelineResult:
    """Define and run the pipeline with the following components:

    - Text Splitter: to split the text into manageable chunks of fixed size
    - Chunk Embedder: to embed the chunks' text
    - Lexical Graph Builder: to build the lexical graph, ie creating the chunk nodes and relationships between them
    - KG writer: save the lexical graph to Neo4j
    """
    pipe = Pipeline()
    # define the components
    pipe.add_component(
        FixedSizeSplitter(chunk_size=200, chunk_overlap=50, approximate=False),
        "splitter",
    )
    pipe.add_component(TextChunkEmbedder(embedder=OpenAIEmbeddings()), "chunk_embedder")
    pipe.add_component(
        LexicalGraphBuilder(lexical_graph_config),
        "lexical_graph_builder",
    )
    pipe.add_component(Neo4jWriter(neo4j_driver), "writer")
    # define the execution order of component
    # and how the output of previous components must be used
    pipe.connect("splitter", "chunk_embedder", input_config={"text_chunks": "splitter"})
    pipe.connect(
        "chunk_embedder",
        "lexical_graph_builder",
        input_config={"text_chunks": "chunk_embedder"},
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
            "text": text,
        },
        "lexical_graph_builder": {
            "document_info": {
                # 'path' can be anything
                "path": "example/lexical_graph_from_text.py"
            },
        },
    }
    # run the pipeline
    return await pipe.run(pipe_inputs)


async def read_chunk_and_perform_entity_extraction(
    neo4j_driver: neo4j.Driver,
    llm: LLMInterface,
    lexical_graph_config: LexicalGraphConfig,
) -> PipelineResult:
    """This is where we define and run the KG builder pipeline, instantiating a few
    components:

    - Neo4j Chunk Reader: to embed the chunks' text
    - Schema Builder: this component takes a list of entities, relationships and
        possible triplets as inputs, validate them and return a schema ready to use
        for the rest of the pipeline
    - LLM Entity Relation Extractor is an LLM-based entity and relation extractor:
        based on the provided schema, the LLM will do its best to identity these
        entities and their relations within the provided text
    - KG writer: once entities and relations are extracted, they can be writen
        to a Neo4j database
    """
    pipe = Pipeline()
    # define the components
    pipe.add_component(Neo4jChunkReader(neo4j_driver), "reader")
    pipe.add_component(SchemaBuilder(), "schema")
    pipe.add_component(
        LLMEntityRelationExtractor(
            llm=llm,
            create_lexical_graph=False,
        ),
        "extractor",
    )
    pipe.add_component(Neo4jWriter(neo4j_driver), "writer")
    # define the execution order of component
    # and how the output of previous components must be used
    pipe.connect("reader", "extractor", input_config={"chunks": "reader"})
    pipe.connect("schema", "extractor", input_config={"schema": "schema"})
    pipe.connect(
        "extractor",
        "writer",
        input_config={"graph": "extractor"},
    )
    # user input:
    # the initial text
    # and the list of entities and relations we are looking for
    pipe_inputs = {
        "reader": {
            "lexical_graph_config": lexical_graph_config,
        },
        "schema": {
            "node_types": [
                NodeType(
                    label="Person",
                    properties=[
                        PropertyType(name="name", type="STRING"),
                        PropertyType(name="place_of_birth", type="STRING"),
                        PropertyType(name="date_of_birth", type="DATE"),
                    ],
                ),
                NodeType(
                    label="Organization",
                    properties=[
                        PropertyType(name="name", type="STRING"),
                        PropertyType(name="country", type="STRING"),
                    ],
                ),
                NodeType(
                    label="Field",
                    properties=[
                        PropertyType(name="name", type="STRING"),
                    ],
                ),
            ],
            "relationship_types": [
                RelationshipType(
                    label="WORKED_ON",
                ),
                RelationshipType(
                    label="WORKED_FOR",
                ),
            ],
            "patterns": [
                ("Person", "WORKED_ON", "Field"),
                ("Person", "WORKED_FOR", "Organization"),
            ],
        },
        "extractor": {
            "lexical_graph_config": lexical_graph_config,
        },
    }
    # run the pipeline
    return await pipe.run(pipe_inputs)


async def main(driver: neo4j.Driver) -> PipelineResult:
    # optional: define some custom node labels for the lexical graph:
    lexical_graph_config = LexicalGraphConfig(
        document_node_label="Book",  # default: "Document"
        chunk_node_label="Chapter",  # default "Chunk"
        chunk_text_property="content",  # default: "text"
    )
    text = """Albert Einstein was a German physicist born in 1879 who
            wrote many groundbreaking papers especially about general relativity
            and quantum mechanics. He worked for many different institutions, including
            the University of Bern in Switzerland and the University of Oxford."""
    llm = OpenAILLM(
        model_name="gpt-4o",
        model_params={
            "max_tokens": 1000,
            "response_format": {"type": "json_object"},
        },
    )
    await build_lexical_graph(driver, lexical_graph_config, text=text)
    res = await read_chunk_and_perform_entity_extraction(
        driver, llm, lexical_graph_config
    )
    await llm.async_client.close()
    return res


if __name__ == "__main__":
    with neo4j.GraphDatabase.driver(
        "bolt://localhost:7687", auth=("neo4j", "password")
    ) as driver:
        print(asyncio.run(main(driver)))
