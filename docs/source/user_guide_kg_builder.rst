.. _user-guide-kg-builder:

User Guide: Knowledge Graph Builder
########################################

This page provides information about how to create a Knowledge Graph from
unstructured data.


******************************
Pipeline structure
******************************

A Knowledge Graph (KG) construction pipeline requires a few components:
- A document parser: extract text from PDFs
- A document chunker: split the text into smaller pieces of text, manageable by the LLM context window (token limit).
- A chunk embeder (optional): compute and store the chunk embeddings
- A schema builder: provide a schema to ground the LLM extracted entities and relations and obtain an easily navigable KG.
- An entity and relation extractor: extract relevant entities and relations from the text.
- A KG writer: write the identified entities and relations to a Neo4j database.


TODO: add drawing

.. code:: python

    from neo4j_genai.components import *


***************************************
Components
***************************************

Below is a list of the different components available in this package, and how to use them.

Document Chunker
========================

Chunk Embeder
===============================

Schema Builder
========================


Entity and Relation Extractor
===============================


Knowledge Graph Writer
===============================
