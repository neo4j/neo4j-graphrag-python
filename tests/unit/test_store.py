from neo4j_genai.core.stores import InMemoryStore
from sympy.testing import pytest


def test_memory_store():
    store = InMemoryStore()
    store.add("key", "value")
    assert store.get("key") == "value"

    with pytest.raises(KeyError):
        store.add("key", "value", overwrite=False)

    assert store.find_all("key") == ["value"]
    assert store.all() == {"key": "value"}
