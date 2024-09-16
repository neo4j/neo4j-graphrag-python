from __future__ import annotations

import asyncio

import neo4j
from langchain_text_splitters import CharacterTextSplitter
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.experimental.components.embedder import TextChunkEmbedder
from neo4j_graphrag.experimental.components.kg_writer import Neo4jWriter
from neo4j_graphrag.experimental.components.lexical_graph import (
    LexicalGraphBuilder,
    LexicalGraphConfig,
)
from neo4j_graphrag.experimental.components.text_splitters.langchain import (
    LangChainTextSplitterAdapter,
)
from neo4j_graphrag.experimental.pipeline import Pipeline
from neo4j_graphrag.experimental.pipeline.pipeline import PipelineResult


async def main(neo4j_driver: neo4j.Driver) -> PipelineResult:
    """This is where we define and run the KG builder pipeline, instantiating a few
    components:
    - Text Splitter: in this example we use a text splitter from the LangChain package
    - Chunk Embedder: to embed the chunks' text
    - Lexical Graph Builder: to build the lexical graph
    - KG writer: save the lexical graph to Neo4j
    """
    pipe = Pipeline()
    # define the components
    pipe.add_component(
        LangChainTextSplitterAdapter(
            # chunk_size=50 for the sake of this demo
            CharacterTextSplitter(chunk_size=50, chunk_overlap=10, separator=".")
        ),
        "splitter",
    )
    pipe.add_component(TextChunkEmbedder(embedder=OpenAIEmbeddings()), "chunk_embedder")
    pipe.add_component(
        LexicalGraphBuilder(LexicalGraphConfig(id_prefix="example")),
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
        input_config={"graph": "lexical_graph_builder"},
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
                "path": "example/lexical_graph_from_text.py"
            }
        },
    }
    # run the pipeline
    return await pipe.run(pipe_inputs)


if __name__ == "__main__":
    with neo4j.GraphDatabase.driver(
        "bolt://localhost:7687", auth=("neo4j", "password")
    ) as driver:
        print(asyncio.run(main(driver)))
