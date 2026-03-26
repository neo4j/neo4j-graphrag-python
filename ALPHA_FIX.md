# Fix: `alpha=0.0` raises `InvalidHybridSearchRankerError`

## File changed

`src/neo4j_graphrag/neo4j_queries.py` line 405

## What was wrong

```python
elif ranker == HybridSearchRanker.LINEAR and alpha:
```

`alpha=0.0` is falsy in Python, so this condition fails and falls through to `raise InvalidHybridSearchRankerError()`. Pure fulltext search (`alpha=0.0`) is impossible with the linear ranker.

## Fix

```python
elif ranker == HybridSearchRanker.LINEAR and alpha is not None:
```

The validator in `types.py` (`HybridSearchModel.validate_alpha`) already correctly accepts `0.0` — the bug was only in the query builder.
