import pytest
from neo4j_genai.experimental.pipeline.stores import InMemoryStore


@pytest.mark.asyncio
async def test_memory_store() -> None:
    store = InMemoryStore()
    await store.add("key", "value")
    res = await store.get("key")
    assert res == "value"

    with pytest.raises(KeyError):
        await store.add("key", "value", overwrite=False)

    assert store.all() == {"key": "value"}
