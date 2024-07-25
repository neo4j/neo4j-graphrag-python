import pytest
from neo4j_genai.core.stores import InMemoryStore


def test_memory_store() -> None:
    store = InMemoryStore()
    store.add("key", "value")
    assert store.get("key") == "value"

    with pytest.raises(KeyError):
        store.add("key", "value", overwrite=False)

    assert store.find_all("key") == ["value"]
    assert store.all() == {"key": "value"}
