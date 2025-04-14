from abc import ABC
from enum import Enum
from typing import Any, Dict, List, Callable, Optional


class ParameterType(str, Enum):
    """Enum for parameter types supported in tool parameters."""

    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    OBJECT = "object"
    ARRAY = "array"


class ToolParameter:
    """Base class for all tool parameters."""

    def __init__(self, description: str, required: bool = False):
        self.description = description
        self.required = required

    def to_dict(self) -> Dict[str, Any]:
        """Convert the parameter to a dictionary format."""
        raise NotImplementedError("Subclasses must implement to_dict")


class StringParameter(ToolParameter):
    """String parameter for tools."""

    def __init__(
        self, description: str, required: bool = False, enum: Optional[List[str]] = None
    ):
        super().__init__(description, required)
        self.enum = enum

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "type": ParameterType.STRING,
            "description": self.description,
        }
        if self.enum:
            result["enum"] = self.enum
        return result


class IntegerParameter(ToolParameter):
    """Integer parameter for tools."""

    def __init__(
        self,
        description: str,
        required: bool = False,
        minimum: Optional[int] = None,
        maximum: Optional[int] = None,
    ):
        super().__init__(description, required)
        self.minimum = minimum
        self.maximum = maximum

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "type": ParameterType.INTEGER,
            "description": self.description,
        }
        if self.minimum is not None:
            result["minimum"] = self.minimum
        if self.maximum is not None:
            result["maximum"] = self.maximum
        return result


class NumberParameter(ToolParameter):
    """Number parameter for tools."""

    def __init__(
        self,
        description: str,
        required: bool = False,
        minimum: Optional[float] = None,
        maximum: Optional[float] = None,
    ):
        super().__init__(description, required)
        self.minimum = minimum
        self.maximum = maximum

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "type": ParameterType.NUMBER,
            "description": self.description,
        }
        if self.minimum is not None:
            result["minimum"] = self.minimum
        if self.maximum is not None:
            result["maximum"] = self.maximum
        return result


class BooleanParameter(ToolParameter):
    """Boolean parameter for tools."""

    def to_dict(self) -> Dict[str, Any]:
        return {"type": ParameterType.BOOLEAN, "description": self.description}


class ObjectParameter(ToolParameter):
    """Object parameter for tools."""

    def __init__(
        self,
        description: str,
        properties: Dict[str, ToolParameter],
        required: bool = False,
        required_properties: Optional[List[str]] = None,
        additional_properties: bool = True,
    ):
        super().__init__(description, required)
        self.properties = properties
        self.required_properties = required_properties or []
        self.additional_properties = additional_properties

    def to_dict(self) -> Dict[str, Any]:
        properties_dict: Dict[str, Any] = {}
        for name, param in self.properties.items():
            properties_dict[name] = param.to_dict()

        result: Dict[str, Any] = {
            "type": ParameterType.OBJECT,
            "description": self.description,
            "properties": properties_dict,
        }

        if self.required_properties:
            result["required"] = self.required_properties

        if not self.additional_properties:
            result["additionalProperties"] = False

        return result


class ArrayParameter(ToolParameter):
    """Array parameter for tools."""

    def __init__(
        self,
        description: str,
        items: ToolParameter,
        required: bool = False,
        min_items: Optional[int] = None,
        max_items: Optional[int] = None,
    ):
        super().__init__(description, required)
        self.items = items
        self.min_items = min_items
        self.max_items = max_items

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "type": ParameterType.ARRAY,
            "description": self.description,
            "items": self.items.to_dict(),
        }

        if self.min_items is not None:
            result["minItems"] = self.min_items

        if self.max_items is not None:
            result["maxItems"] = self.max_items

        return result


class Tool(ABC):
    """Abstract base class defining the interface for all tools in the neo4j-graphrag library."""

    def __init__(
        self,
        name: str,
        description: str,
        parameters: ObjectParameter,
        execute_func: Callable[..., Any],
    ):
        self._name = name
        self._description = description
        self._parameters = parameters
        self._execute_func = execute_func

    def get_name(self) -> str:
        """Get the name of the tool.

        Returns:
            str: Name of the tool.
        """
        return self._name

    def get_description(self) -> str:
        """Get a detailed description of what the tool does.

        Returns:
            str: Description of the tool.
        """
        return self._description

    def get_parameters(self) -> Dict[str, Any]:
        """Get the parameters the tool accepts in a dictionary format suitable for LLM providers.

        Returns:
            Dict[str, Any]: Dictionary containing parameter schema information.
        """
        return self._parameters.to_dict()

    def execute(self, query: str, **kwargs: Any) -> Any:
        """Execute the tool with the given query and additional parameters.

        Args:
            query (str): The query or input for the tool to process.
            **kwargs (Any): Additional parameters for the tool.

        Returns:
            Any: The result of the tool execution.
        """
        return self._execute_func(query, **kwargs)
