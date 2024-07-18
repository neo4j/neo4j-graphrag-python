import abc
from typing import Any

from jsonpath_ng import parse


class Store(abc.ABC):
    """An interface to save component outputs"""

    @abc.abstractmethod
    def add(self, key: str, value: Any, overwrite: bool = True) -> None:
        """
        Args:
            key (str): The key to access the data.
            value (Any): The value to store in the data.
            overwrite (bool): Whether to overwrite existing data.
                If overwrite is False and the key already exists
                in the store, an exception is raised.

        Raises:
            KeyError: If the key already exists in the store and overwrite is False.
        """
        pass

    @abc.abstractmethod
    def get(self, key: str) -> Any:
        """Retrieve value for `key`.
        If key not found, returns None.
        """
        pass

    @abc.abstractmethod
    def find_all(self, pattern: str) -> list[Any]:
        """Find all values whose key matches pattern"""
        pass

    def all(self) -> dict[str, Any]:
        """Return all stored data
        Might not be relevant to implement
        in all subclasses, that's why it is
        not marked as abstract.
        """
        raise NotImplementedError()


class InMemoryStore(Store):
    """Simple in-memory store.
    Saves each component's results in a _data dict."""

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}

    def add(self, key: str, value: Any, overwrite: bool = True) -> None:
        if (not overwrite) and key in self._data:
            raise KeyError(f"{key} already exists")
        self._data[key] = value

    def get(self, key: str) -> Any:
        return self._data.get(key)

    def find_all(self, pattern: str) -> list[Any]:
        jsonpath_expr = parse(pattern)
        # input_component, param = mapping.split(".")
        # value = self._results[input_component][param]
        value = [match.value for match in jsonpath_expr.find(self._data)]
        return value

    def all(self) -> dict[str, Any]:
        return self._data
