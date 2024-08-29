.. _user-guide-kg-builder:

User Guide: Knowledge Graph Builder
###################################


This page provides information about how to create a Knowledge Graph from
unstructured data.

.. warning::

    This feature is still experimental. API changes and bug fixes are expected.

    It is not recommended to use it in production yet.


******************
Pipeline Structure
******************

A Knowledge Graph (KG) construction pipeline requires a few components:

- **Document parser**: extract text from files (PDFs, ...).
- **Document chunker**: split the text into smaller pieces of text, manageable by the LLM context window (token limit).
- **Chunk embedder** (optional): compute the chunk embeddings.
- **Schema builder**: provide a schema to ground the LLM extracted entities and relations and obtain an easily navigable KG.
- **Entity and relation extractor**: extract relevant entities and relations from the text.
- **Knowledge Graph writer**: save the identified entities and relations.

.. image:: images/kg_builder_pipeline.png
  :alt: KG Builder pipeline


This package contains the interface and implementations for each of these components, which are detailed in the following sections.

To see an end-to-end example of a Knowledge Graph construction pipeline,
refer to `this example <https://github.com/neo4j/neo4j-genai-python/blob/main/examples/pipeline/kg_builder.py>`_.

**********************************
Knowledge Graph Builder Components
**********************************

Below is a list of the different components available in this package and how to use them.

Each of these components can be run individually:

.. code:: python

    import asyncio
    from neo4j_genai.experimental.components.pdf_loader import PdfLoader
    my_component = PdfLoader()
    asyncio.run(my_component.run("my_file.pdf"))


They can also be used within a pipeline:

.. code:: python

    from neo4j_genai.experimental.pipeline import Pipeline
    from neo4j_genai.experimental.components.pdf_loader import PdfLoader
    pipeline = Pipeline()
    my_component = PdfLoader()
    pipeline.add_component(my_component, "component_name")


Document Parser
===============

Document parsers start from a file path and return the text extracted from this file.

This package currently supports text extraction from PDFs:

.. code:: python

    from pathlib import Path
    from neo4j_genai.experimental.components.pdf_loader import PdfLoader

    loader = PdfLoader()
    loader.run(path=Path("my_file.pdf"))

To implement your own loader, use the `DataLoader` interface:

.. code:: python

    from pathlib import Path
    from neo4j_genai.experimental.components.pdf_loader import DataLoader, PdfDocument

    class MyDataLoader(DataLoader):
        async def run(self, path: Path) -> PdfDocument:
            # process file in `path`
            return PdfDocument(text="text")



Document Splitter
=================

Document splitters, as the name indicate, split documents into smaller chunks
that can be processed within the LLM token limits. Wrappers for LangChain and LlamaIndex
text splitters are included in this package:


.. code:: python

    from langchain_text_splitters import CharacterTextSplitter
    from neo4j_genai.experimental.components.text_splitters.langchain import LangChainTextSplitterAdapter
    splitter = LangChainTextSplitterAdapter(
        CharacterTextSplitter(chunk_size=500, chunk_overlap=100, separator=".")
    )
    splitter.run(text="Hello World. Life is beautiful.")


Also see :ref:`langchaintextsplitteradapter` and :ref:`llamaindextextsplitteradapter`.

To implement a custom text splitter, the `TextSplitter` interface can be used:

.. code:: python

    from neo4j_genai.experimental.components.text_splitters.base import TextSplitter
    from neo4j_genai.experimental.components.types import TextChunks, TextChunk


    class MyTextSplitter(TextSplitter):

        def __init__(self, separator: str = ".") -> None:
            self.separator = separator

        async def run(self, text: str) -> TextChunks:
             return TextChunks(
                 chunks=[
                     TextChunk(text=text_chunk)
                     for text_chunk in text.split(self.separator)
                 ]
             )


Chunk Embedder
==============

In order to embed the chunks' texts (to be used in vector search RAG), one can use the
`TextChunkEmbedder` component, which rely on the :ref:`Embedder` interface.

Example usage:

.. code:: python

    from neo4j_genai.experimental.components.embedder import TextChunkEmbedder
    from neo4j_genai.embeddings.openai import OpenAIEmbeddings
    text_chunk_embedder = TextChunkEmbedder(embedder=OpenAIEmbeddings())
    text_chunk_embedder.run(text_chunks=TextChunks(chunks=[TextChunk(text="my_text")]))

.. note::

    To use OpenAI (embedding or LLM), the `OPENAI_API_KEY` must be in the env vars, for instance using:

    .. code:: python

        import os
        os.environ["OPENAI_API_KEY"] = "sk-..."


If OpenAI is not an option, see :ref:`embedders` to learn how to use sentence-transformers or create your own embedder.

The embeddings are added to each chunk metadata, and will be saved as a Chunk node property in the graph if
`create_lexical_graph` is enabled in the `EntityRelationExtractor` (keep reading).


Schema Builder
==============

The schema is used to try and ground the LLM to a list of possible entities and relations of interest.
So far, schema must be manually created by specifying:

- **Entities** the LLM should look for in the text, including their properties (name and type).
- **Relations** of interest between these entities, including the relation properties (name and type).
- **Triplets** to define the start (source) and end (target) entity types for each relation.

Here is a code block illustrating these concepts:

.. code:: python

    from neo4j_genai.experimental.components.schema import (
        SchemaBuilder,
        SchemaEntity,
        SchemaProperty,
        SchemaRelation,
    )

    schema_builder = SchemaBuilder()

    schema_builder.run(
        entities=[
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
        ],
        relations=[
            SchemaRelation(
                label="WORKED_ON",
            ),
            SchemaRelation(
                label="WORKED_FOR",
            ),
        ],
        possible_schema=[
            ("Person", "WORKED_ON", "Field"),
            ("Person", "WORKED_FOR", "Organization"),
        ],
    )

After validation, this schema is saved in a `SchemaConfig` object, whose dict representation is passed
to the LLM.


Entity and Relation Extractor
=============================

This component is responsible for extracting the relevant entities and relationships from each text chunk,
using the schema as guideline.

This package contains an LLM-based entity and relationships extractor: `LLMEntityRelationExtractor`.
It can be used in this way:

.. code:: python

    from neo4j_genai.experimental.components.entity_relation_extractor import (
        LLMEntityRelationExtractor,
    )
    from neo4j_genai.llm import OpenAILLM

    extractor = LLMEntityRelationExtractor(
        llm=OpenAILLM(
            model_name="gpt-4o",
            model_params={
                "max_tokens": 1000,
                "response_format": {"type": "json_object"},
            },
        )
    )

.. warning::

    The `LLMEntityRelationExtractor` works better if `"response_format": {"type": "json_object"}` is in the model parameters.

The LLM to use can be customized, the only constraint is that it obeys the :ref:`LLMInterface <llminterface>`.

Error Behaviour
---------------

By default, if the extraction fails for one chunk, it will be ignored and the non-failing chunks will be saved.
This behaviour can be changed by using the `on_error` flag in the `LLMEntityRelationExtractor` constructor:

.. code:: python

    from neo4j_genai.experimental.components.entity_relation_extractor import (
        LLMEntityRelationExtractor,
        OnError,
    )

    extractor = LLMEntityRelationExtractor(
        llm=OpenAILLM(
            model_name="gpt-4o",
            model_params={
                "max_tokens": 1000,
                "response_format": {"type": "json_object"},
            },
        ),
        on_error=OnError.RAISE,
    )

In this scenario, any failing chunk will make the whole pipeline fail (for all chunks), and no data
will be saved to Neo4j.


Lexical Graph
-------------

By default, the `LLMEntityRelationExtractor` adds some extra nodes and relationships to the extracted graph:

- `Chunk` nodes: represent the text chunks. They have a `text` property and, if computed, an `embedding` property.
- `NEXT_CHUNK` relationships between one chunk node and the next one in the document. It can be used to enhance the context in a RAG application.
- `FROM_CHUNK` relationship between any extracted entity and the chunk it has been identified into.

If this 'lexical graph' is not desired, set the `created_lexical_graph` to `False` in the extractor constructor:

.. code:: python

    extractor = LLMEntityRelationExtractor(
        llm=....,
        create_lexical_graph=False,
    )


Customizing the Prompt
----------------------

The default prompt uses the :ref:`erextractiontemplate`. It is possible to provide a custom prompt as string:

.. code:: python

    extractor = LLMEntityRelationExtractor(
        llm=....,
        prompt="this is my prompt",
    )

The following variables can be used in the prompt:

- `text` (str): the text to be analyzed.
- `schema` (str): the graph schema to be used.
- `examples` (str): examples for few-shot learning.


Subclassing the EntityRelationExtractor
---------------------------------------

If more customization is needed, it is possible to subclass the `EntityRelationExtractor` interface:

.. code:: python

    from pydantic import validate_call
    from neo4j_genai.experimental.components.entity_relation_extractor import EntityRelationExtractor
    from neo4j_genai.experimental.components.schema import SchemaConfig
    from neo4j_genai.experimental.components.types import (
        Neo4jGraph,
        Neo4jNode,
        Neo4jRelationship,
        TextChunks,
    )

    class MyExtractor(EntityRelationExtractor):

    @validate_call
    async def run(self, chunks: TextChunks, **kwargs: Any) -> Neo4jGraph:
        return Neo4jGraph(
            nodes=[
                Neo4jNode(id="0", label="Person", properties={"name": "A. Einstein"}),
                Neo4jNode(id="1", label="Concept", properties={"name": "Theory of relativity"}),
            ],
            relationships=[
                Neo4jRelationship(type="PROPOSED_BY", start_node_id="1", end_node_id="0", properties={"year": 1915})
            ],
        )


See :ref:`entityrelationextractor`.


Knowledge Graph Writer
======================

KG writer are used to save the results of the `EntityRelationExtractor`.
The main implementation is the `Neo4jWriter` that will write nodes and relationships
to a Neo4j database:

.. code:: python

    import neo4j
    from neo4j_genai.experimental.components.kg_writer import Neo4jWriter
    from neo4j_genai.experimental.components.types import Neo4jGraph

    with neo4j.GraphDatabase.driver(
        "bolt://localhost:7687", auth=("neo4j", "password")
    ) as driver:
        writer = Neo4jWriter(driver)
        graph = Neo4jGraph(nodes=[], relationships=[])
        asyncio.run(writer.run())

See :ref:`neo4jgraph` for the description of the input type.

It is possible to create a custom writer using the `KGWriter` interface:

.. code:: python

    import json
    from pydantic import validate_call
    from neo4j_genai.experimental.components.kg_writer import KGWriter

    class JsonWriter(KGWriter):

        def __init__(self, file_name: str) -> None:
            self.file_name = file_name

        @validate_call
        async def run(self, graph: Neo4jGraph) -> KGWriterModel:
            try:
                with open(self.file_name, "w") as f:
                    json.dump(graph.model_dump(), f, indent=2)
                return KGWriterModel(status="SUCCESS")
            except Exception:
                return KGWriterModel(status="FAILURE")


.. note::

    The `validate_call` decorator is required when the input parameter contain a `pydantic` model.


See :ref:`kgwritermodel` and :ref:`kgwriter` in API reference.
