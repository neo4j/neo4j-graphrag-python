"""Minimal example of the dataflow pipeline: source -> stages -> sink.

A pipeline is a plain description of the work: `Pipeline.from_source(...)`
followed by operators such as `map`, `filter` and `flat_map`. Nothing runs until
the pipeline is consumed with `collect()`, `to_sink()` or by iterating over it.

This example shows the modularity of the design: the source, each stage and the
sink are independent pieces, and any of them can be swapped without touching
the others.

It needs no database and no API key:

    python examples/pipeline/01_basic_pipeline.py

NB: this is *not* `neo4j_graphrag.experimental.pipeline.Pipeline` (the
component/`connect` based one used by `SimpleKGPipeline`).
"""

from collections.abc import Iterable

from neo4j_graphrag.pipeline import Pipeline, Sink, Source


class InMemoryDocuments(Source[str]):
    """A source: anything that can `read()` an iterable of elements."""

    def __init__(self, documents: list[str]) -> None:
        self.documents = documents

    def read(self) -> Iterable[str]:
        return self.documents


class PrintSink(Sink[str]):
    """A sink: receives every element that comes out of the pipeline."""

    def write(self, element: str) -> None:
        print(f"-> {element}")


# Stages are plain functions: easy to test, easy to replace.
def normalize(text: str) -> str:
    return " ".join(text.split()).lower()


def is_not_empty(text: str) -> bool:
    return bool(text)


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in text.split(".") if s.strip()]


documents = InMemoryDocuments(
    [
        "  Alice WORKS at Acme.   Bob works at Initech. ",
        "",
        "Carol lives in Paris.",
    ]
)

# 1. Define the pipeline. This only builds the operator graph, nothing runs yet.
pipeline = (
    Pipeline.from_source(documents)
    .map(normalize)
    .filter(is_not_empty)
    .flat_map(split_sentences)  # one document -> many sentences
)

# 2. Run it and get a list back...
print(pipeline.collect())

# 3. ...or stream the results into a sink instead.
pipeline.to_sink(PrintSink())

# 4. Modularity: a pipeline is reusable, so extending it is just adding stages.
#    `pipeline` itself is unchanged.
print(pipeline.map(str.upper).take(2).collect())
