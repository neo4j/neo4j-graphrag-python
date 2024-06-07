# Neo4j GenAI package for Python

This repository contains the official Neo4j GenAI features for Python.

The purpose of this package is to provide a first party package to developers,
where Neo4j can guarantee long term commitment and maintenance as well as being
fast to ship new features and high performing patterns and methods.

Docs are coming soon!

# Usage

## Installation

This package requires Python (>=3.8.1).

To install the latest stable version, use:

```shell
pip install neo4j-genai
```

## Examples

While the library has more retrievers than shown here, the following examples should be able to get you started.

### Performing a similarity search

Assumption: Neo4j running with populated vector index in place.

```python
from neo4j import GraphDatabase
from neo4j_genai import VectorRetriever
from langchain_openai import OpenAIEmbeddings

URI = "neo4j://localhost:7687"
AUTH = ("neo4j", "password")

INDEX_NAME = "embedding-name"

# Connect to Neo4j database
driver = GraphDatabase.driver(URI, auth=AUTH)

# Create Embedder object
embedder = OpenAIEmbeddings(model="text-embedding-3-large")

# Initialize the retriever
retriever = VectorRetriever(driver, INDEX_NAME, embedder)

# Run the similarity search
query_text = "How do I do similarity search in Neo4j?"
response = retriever.search(query_text=query_text, top_k=5)
```

### Creating a vector index

When creating a vector index, make sure you match the number of dimensions in the index with the number of dimensions the embeddings have.

Assumption: Neo4j running

```python
from neo4j import GraphDatabase
from neo4j_genai.indexes import create_vector_index

URI = "neo4j://localhost:7687"
AUTH = ("neo4j", "password")

INDEX_NAME = "chunk-index"

# Connect to Neo4j database
driver = GraphDatabase.driver(URI, auth=AUTH)

# Creating the index
create_vector_index(
    driver,
    INDEX_NAME,
    label="Document",
    property="textProperty",
    dimensions=1536,
    similarity_fn="euclidean",
)

```

### Populating the Neo4j Vector Index

This library does not write to the database, that is up to you.
See below for how to write using Cypher via the Neo4j driver.

Assumption: Neo4j running with a defined vector index

```python
from neo4j import GraphDatabase
from random import random

URI = "neo4j://localhost:7687"
AUTH = ("neo4j", "password")

# Connect to Neo4j database
driver = GraphDatabase.driver(URI, auth=AUTH)

# Upsert the vector
vector = [random() for _ in range(DIMENSION)]
insert_query = (
    "MERGE (n:Document {id: $id})"
    "WITH n "
    "CALL db.create.setNodeVectorProperty(n, 'textProperty', $vector)"
    "RETURN n"
)
parameters = {
    "id": 0,
    "vector": vector,
}
driver.execute_query(insert_query, parameters)
```

# Development

## Install dependencies

```bash
poetry install
```

## Getting started

### Issues

If you have a bug to report or feature to request, first
[search to see if an issue already exists](https://docs.github.com/en/github/searching-for-information-on-github/searching-on-github/searching-issues-and-pull-requests#search-by-the-title-body-or-comments).
If a related issue doesn't exist, please raise a new issue using the relevant
[issue form](https://github.com/neo4j/neo4j-genai-python/issues/new/choose).

If you're a Neo4j Enterprise customer, you can also reach out to [Customer Support](http://support.neo4j.com/).

If you don't have a bug to report or feature request, but you need a hand with
the library; community support is available via [Neo4j Online Community](https://community.neo4j.com/)
and/or [Discord](https://discord.gg/neo4j).

### Make changes

1. Fork the repository.
2. Install Python and Poetry.
3. Create a working branch from `main` and start with your changes!

### Pull request

When you're finished with your changes, create a pull request, also known as a PR.

-   Ensure that you have [signed the CLA](https://neo4j.com/developer/contributing-code/#sign-cla).
-   Ensure that the base of your PR is set to `main`.
-   Don't forget to [link your PR to an issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/linking-a-pull-request-to-an-issue)
    if you are solving one.
-   Enable the checkbox to [allow maintainer edits](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/allowing-changes-to-a-pull-request-branch-created-from-a-fork)
    so that maintainers can make any necessary tweaks and update your branch for merge.
-   Reviewers may ask for changes to be made before a PR can be merged, either using
    [suggested changes](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/reviewing-changes-in-pull-requests/incorporating-feedback-in-your-pull-request)
    or normal pull request comments. You can apply suggested changes directly through
    the UI, and any other changes can be made in your fork and committed to the PR branch.
-   As you update your PR and apply changes, mark each conversation as [resolved](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/commenting-on-a-pull-request#resolving-conversations).
-   Update the `CHANGELOG.md` if you have made significant changes to the project, these include:
    - Major changes:
      - New features
      - Bug fixes with high impact
      - Breaking changes
    - Minor changes:
      - Documentation improvements
      - Code refactoring without functional impact
      - Minor bug fixes
- Keep `CHANGELOG.md` changes brief and focus on the most important changes.

### Updating the `CHANGELOG.md`
1. When opening a PR, you can generate an edit suggestion by commenting on the GitHub PR [using CodiumAI](https://github.com/CodiumAI-Agent):

```
@CodiumAI-Agent /update_changelog
```

2. Use this as a suggestion and update the `CHANGELOG.md` content under 'Next'.
3. Commit the changes.

## Run tests

### Unit tests

This should run out of the box once the dependencies are installed.

```bash
poetry run pytest tests/unit
```

### E2E tests

To run e2e tests you'd need to have some services running locally:

-   neo4j
-   weaviate
-   weaviate-text2vec-transformers

The easiest way to get it up and running is via Docker compose:

```bash
docker compose -f tests/e2e/docker-compose.yml up
```

_(pro tip: if you suspect something in the databases are cached, run `docker compose -f tests/e2e/docker-compose.yml down` to remove them completely)_

Once the services are running, execute the following command to run the e2e tests.

```bash
poetry run pytest tests/e2e
```

## Further information

-   [The official Neo4j Python driver](https://github.com/neo4j/neo4j-python-driver)
-   [Neo4j GenAI integrations](https://neo4j.com/docs/cypher-manual/current/genai-integrations/)
