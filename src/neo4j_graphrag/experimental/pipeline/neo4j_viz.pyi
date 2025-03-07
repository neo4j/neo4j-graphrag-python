from typing import Any, Dict, List, Optional, Union

class Node:
    id: Union[str, int]
    caption: Optional[str] = None
    size: Optional[float] = None
    properties: Optional[Dict[str, Any]] = None

    def __init__(
        self,
        id: Union[str, int],
        caption: Optional[str] = None,
        size: Optional[float] = None,
        properties: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None: ...

class Relationship:
    source: Union[str, int]
    target: Union[str, int]
    caption: Optional[str] = None
    properties: Optional[Dict[str, Any]] = None

    def __init__(
        self,
        source: Union[str, int],
        target: Union[str, int],
        caption: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None: ...

class VisualizationGraph:
    nodes: List[Node]
    relationships: List[Relationship]

    def __init__(
        self, nodes: List[Node], relationships: List[Relationship]
    ) -> None: ...
