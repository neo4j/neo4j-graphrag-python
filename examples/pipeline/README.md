# Dataflow pipeline examples

Minimal examples of `neo4j_graphrag.pipeline.Pipeline`, a lazily-evaluated
dataflow pipeline: `Pipeline.from_source(...).map(...).filter(...).to_sink(...)`.

> This is **not** `neo4j_graphrag.experimental.pipeline.Pipeline`, the
> component-graph pipeline behind `SimpleKGPipeline`.

Run from the repository root, e.g. `python examples/pipeline/01_basic_pipeline.py`.

| Example | Shows | Needs |
|---------|-------|-------|
| [01_basic_pipeline.py](01_basic_pipeline.py) | Source, stages and sink as independent, swappable pieces | nothing |
| [02_configuring_stages.py](02_configuring_stages.py) | Configuring stages: `partial`, callable classes, `grouped`, `take`/`skip`/`filter`, `map_async_chunked`, `reduce` | nothing |
| [03_error_handling_and_observers.py](03_error_handling_and_observers.py) | `map_safe`, `Ok`/`Err`, `filter_ok`, `on_error`, `LoggingStageObserver` | nothing |
| [04_kg_components.py](04_kg_components.py) | Composing `FixedSizeSplitter`, an extractor and a KG writer | nothing by default; Neo4j to write the graph (set `NEO4J_URI`) |
