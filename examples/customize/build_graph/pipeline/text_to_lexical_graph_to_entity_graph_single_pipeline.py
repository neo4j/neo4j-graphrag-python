"""In this example, we set up a single pipeline with two Neo4j writers:
one for creating the lexical graph (Document and Chunks)
and another for creating the entity graph (entities and relations derived from the text).
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
from neo4j_graphrag.experimental.components.schema import (
    SchemaBuilder,
    SchemaEntity,
    SchemaProperty,
    SchemaRelation,
)
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import (
    FixedSizeSplitter,
)
from neo4j_graphrag.experimental.components.types import LexicalGraphConfig
from neo4j_graphrag.experimental.pipeline import Pipeline
from neo4j_graphrag.experimental.pipeline.pipeline import PipelineResult
from neo4j_graphrag.llm import LLMInterface, OpenAILLM

import neo4j


async def define_and_run_pipeline(
    neo4j_driver: neo4j.Driver,
    llm: LLMInterface,
    lexical_graph_config: LexicalGraphConfig,
    text: str,
) -> PipelineResult:
    """Define and run the pipeline with the following components:

    - Text Splitter: to split the text into manageable chunks of fixed size
    - Chunk Embedder: to embed the chunks' text
    - Lexical Graph Builder: to build the lexical graph, ie creating the chunk nodes and relationships between them
    - LG KG writer: save the lexical graph to Neo4j

    - Schema Builder: this component takes a list of entities, relationships and
        possible triplets as inputs, validate them and return a schema ready to use
        for the rest of the pipeline
    - LLM Entity Relation Extractor is an LLM-based entity and relation extractor:
        based on the provided schema, the LLM will do its best to identity these
        entities and their relations within the provided text
    - EG KG writer: once entities and relations are extracted, they can be writen
        to a Neo4j database

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
    pipe.add_component(Neo4jWriter(neo4j_driver), "lg_writer")
    pipe.add_component(SchemaBuilder(), "schema")
    pipe.add_component(
        LLMEntityRelationExtractor(
            llm=llm,
            create_lexical_graph=False,
        ),
        "extractor",
    )
    pipe.add_component(Neo4jWriter(neo4j_driver), "eg_writer")
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
        "lg_writer",
        input_config={
            "graph": "lexical_graph_builder.graph",
            "lexical_graph_config": "lexical_graph_builder.config",
        },
    )
    # define the execution order of component
    # and how the output of previous components must be used
    pipe.connect(
        "chunk_embedder", "extractor", input_config={"chunks": "chunk_embedder"}
    )
    pipe.connect("schema", "extractor", input_config={"schema": "schema"})
    pipe.connect(
        "extractor",
        "eg_writer",
        input_config={"graph": "extractor"},
    )
    # make sure the lexical graph is created before creating the entity graph:
    pipe.connect("lg_writer", "eg_writer", {})
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
        "schema": {
            "entities": [
                SchemaEntity(
                    label="Person",
                    properties=[
                        SchemaProperty(name="name", type="STRING"),
                        SchemaProperty(name="place_of_birth", type="STRING"),
                        SchemaProperty(name="date_of_birth", type="DATE"),
                    ],
                ),
                SchemaEntity(
                    label="Organization",
                    properties=[
                        SchemaProperty(name="name", type="STRING"),
                        SchemaProperty(name="country", type="STRING"),
                    ],
                ),
                SchemaEntity(
                    label="Field",
                    properties=[
                        SchemaProperty(name="name", type="STRING"),
                    ],
                ),
            ],
            "relations": [
                SchemaRelation(
                    label="WORKED_ON",
                ),
                SchemaRelation(
                    label="WORKED_FOR",
                ),
            ],
            "potential_schema": [
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
        chunk_node_label="TextPart",
        document_node_label="Text",
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
    res = await define_and_run_pipeline(
        driver,
        llm,
        lexical_graph_config,
        text,
    )
    await llm.async_client.close()
    return res


if __name__ == "__main__":
    with neo4j.GraphDatabase.driver(
        "bolt://localhost:7687", auth=("neo4j", "password")
    ) as driver:
        print(asyncio.run(main(driver)))
