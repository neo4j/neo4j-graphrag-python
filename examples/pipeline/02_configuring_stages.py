"""How to configure the stages of a dataflow pipeline.

Every operator takes a function, so a stage is configured by choosing *which*
function to give it:

* `functools.partial` binds parameters to a plain function,
* a callable class holds configuration (and state, such as a client) in its
  constructor,
* operator arguments (`grouped(size)`, `take(n)`, `map_batch_size`, `label`)
  tune how the stream is shaped and how concurrently async stages run.

It needs no database and no API key:

    python examples/pipeline/02_configuring_stages.py
"""

import asyncio
from functools import partial

from neo4j_graphrag.pipeline import Pipeline


# 1. A function configured with functools.partial -----------------------------
def truncate(text: str, max_chars: int) -> str:
    return text if len(text) <= max_chars else text[:max_chars] + "…"


# 2. A callable class holding its own configuration ---------------------------
class Prefixer:
    def __init__(self, prefix: str) -> None:
        self.prefix = prefix

    def __call__(self, text: str) -> str:
        return f"{self.prefix}{text}"


# 3. An async stage, e.g. a call to an LLM or an embedder ---------------------
async def fake_embed(text: str) -> list[float]:
    await asyncio.sleep(0.01)  # stands in for network latency
    return [float(len(text))]


texts = [f"chunk number {i} with some text" for i in range(10)]

pipeline = (
    Pipeline(texts)
    .skip(1)  # drop the first element
    .filter(lambda t: "7" not in t)  # drop elements that fail the predicate
    .map(partial(truncate, max_chars=14), label="truncate")
    .map(Prefixer("> "), label="prefix")
    .take(6)
)
print(pipeline.collect())

# 4. Same pipeline, a different configuration: only the arguments change.
shorter = (
    Pipeline(texts)
    .map(partial(truncate, max_chars=5), label="truncate")
    .map(Prefixer("# "), label="prefix")
    .take(3)
)
print(shorter.collect())

# 5. Batching and concurrency -------------------------------------------------
# `grouped` batches elements (useful for bulk writes); `map_async_chunked`
# runs up to `map_batch_size` async calls concurrently.
embedded = (
    Pipeline(texts)
    .map_async_chunked(fake_embed, map_batch_size=5, label="embed")
    .grouped(4, label="batch")
    .collect()
)
print([len(batch) for batch in embedded])  # [4, 4, 2]

# 6. Aggregation: `reduce` folds the stream into a single value.
total_chars = Pipeline(texts).map(len).reduce(0, lambda acc, n: acc + n).collect()
print(total_chars)
