"""Error handling and observability in a dataflow pipeline.

By default an exception in a stage aborts the whole run. The `*_safe` operators
instead capture a failure for that one element as an `Err` and let the stream
carry on, so a single bad document does not lose the others.

A `StageObserver` can be attached to watch items flow through each stage. For
a deeper walk-through see `examples/pipeline_observability.ipynb`.

It needs no database and no API key:

    python examples/pipeline/03_error_handling_and_observers.py
"""

import logging

from neo4j_graphrag.pipeline import (
    Err,
    LocalInterpreter,
    LoggingStageObserver,
    Pipeline,
)


def parse_year(raw: str) -> int:
    return int(raw)  # raises ValueError on bad input


inputs = ["1998", "2004", "not-a-year", "2021"]

# 1. Without protection, one bad element aborts the run.
try:
    Pipeline(inputs).map(parse_year).collect()
except ValueError as e:
    print(f"aborted: {e}")

# 2. `map_safe` wraps each element in Ok / Err; nothing is lost.
for result in Pipeline(inputs).map_safe(parse_year).collect():
    print(result)

# 3. `filter_ok` silently drops the failures and unwraps the successes...
print(Pipeline(inputs).map_safe(parse_year).filter_ok().collect())

# 4. ...while `on_error` lets you handle each failure (log it, count it, ...).
failures: list[Err] = []
valid_years = (
    Pipeline(inputs)
    .map_safe(parse_year, label="parse")
    .map_ok(lambda year: year + 1)  # only runs on Ok values
    .on_error(failures.append)
    .collect()
)
print(valid_years, f"({len(failures)} failure)")

# 5. Observers: log every stage boundary, item and failure.
logging.basicConfig(level=logging.INFO, format="%(message)s")
observed = Pipeline(inputs).map_safe(parse_year, label="parse").filter_ok()
interpreter = LocalInterpreter(observers=[LoggingStageObserver()])
print(list(interpreter.evaluate(observed)))
