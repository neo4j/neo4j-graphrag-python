# MERGE/Dedup Changes — Not Yet Upstream

## Status

These changes exist on the `bedrock-embeddings` branch (commit `e548410`) but were **not merged into upstream/main**. They need a separate PR.

## Problem

When processing documents with `SimpleKGPipeline`, the same entity (e.g., "Apple Inc.") extracted from multiple chunks causes `CREATE` to produce duplicate nodes. If a uniqueness constraint exists, the write fails with `IndexEntryConflictException`.

## Solution

Replace `CREATE` with `apoc.merge.node` in `Neo4jWriter` so nodes are merged on their primary label + an identifying property (default: `name`).

### Files Changed

1. **`src/neo4j_graphrag/neo4j_queries.py`** — Added `upsert_node_query_merge()` function using `apoc.merge.node` with dynamic label support. Merges on primary label only (not `__Entity__`/`__KGBuilder__`) so pre-existing nodes are matched.

2. **`src/neo4j_graphrag/experimental/components/kg_writer.py`** — Added `use_merge` (default `True`) and `merge_property` (default `"name"`) params to `Neo4jWriter`. Added `_validate_merge_property()` to separate nodes into:
   - Entity nodes with merge_property → MERGE
   - Lexical nodes (Chunk, Document) → CREATE (always unique)
   - Entity nodes missing merge_property → skipped with warning

3. **`tests/unit/experimental/components/test_kg_writer.py`** — ~400 lines of tests for merge query generation, custom merge properties, validation, and metadata reporting.

### Key API

```python
# Default: MERGE on 'name' (recommended)
writer = Neo4jWriter(driver=driver)

# Custom merge key
writer = Neo4jWriter(driver=driver, merge_property="id")

# Old CREATE behavior
writer = Neo4jWriter(driver=driver, use_merge=False)
```

### Metadata returned

```python
result.metadata = {
    "node_count": 100,
    "nodes_created": 95,
    "nodes_skipped_missing_merge_property": 5,
    "relationship_count": 50,
}
```

## Conflicts with Upstream

As of upstream/main (`bf50307`), these files have diverged:
- `neo4j_queries.py` — upstream added `support_dynamic_labels` param to `upsert_node_query()`
- `kg_writer.py` — upstream refactored metadata to use `_graph_stats()` helper and added `ParquetWriter`
- `test_kg_writer.py` — upstream added `is_version_5_24_or_above` checks and `ParquetWriter` tests

When creating the PR, rebase onto current upstream/main and reconcile:
- Add `support_dynamic_labels` param to `upsert_node_query_merge()` as well
- Use `_graph_stats()` pattern for metadata, extending it with merge-specific fields
- Keep upstream's `ParquetWriter` tests alongside merge tests

## Full Design Doc

See `CREATE_MERGE.md` in this branch for the complete technical design document.
