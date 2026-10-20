# Dataflow pipeline examples

Minimal examples of `neo4j_graphrag.pipeline.Pipeline`, a lazily-evaluated
dataflow pipeline: `Pipeline.from_source(...).map(...).filter(...).to_sink(...)`.

> This is **not** `neo4j_graphrag.experimental.pipeline.Pipeline`, the
> component-graph pipeline behind `SimpleKGPipeline`.

Run from the repository root, e.g. `python examples/pipeline/01_basic_pipeline.py`.

| Example | Shows | Needs |
|---------|-------|-------|
| [01_basic_pipeline.py](01_basic_pipeline.py) | Source, stages and sink as independent, swappable pieces | nothing |
