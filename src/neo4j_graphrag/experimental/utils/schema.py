from typing import Any, Union

try:
    from neo4j_viz import VisualizationGraph, Node, Relationship
except ImportError:
    VisualizationGraph = Node = Relationship = None  # type: ignore

from neo4j_graphrag.experimental.components.schema import (
    GraphSchema,
    NodeType,
    PropertyType,
)


def schema_visualization(
    schema: Union[dict[str, Any], GraphSchema],
) -> VisualizationGraph:
    """Helper function to visualize a GraphSchema using the neo4j-viz library.

    Usage:

    .. code:: python

        VG = schema_visualization(schema)
        html = VG.render()

        # in Jupyter:
        display(html)

        # to save the generated HTML
        with open("my_schema.html", "w") as f:
            f.write(html.data)
    """
    if VisualizationGraph is None:
        raise ImportError(
            "Please install neo4j-viz to use the graph schema visualization feature: pip install neo4j-viz"
        )

    schema_object = GraphSchema.model_validate(schema)

    def _format_property_name(p: PropertyType) -> str:
        """

        Args:
            p (PropertyType): the property to be formatted

        Returns:
            str: the property name, suffixed with '*' if the property is required

        """
        return p.name + ("*" if p.required else "")

    def _relationship_properties(rel_type: str) -> dict[str, str]:
        """Returns a dict {prop_name: prop_type} for all relationship properties.

        Args:
            rel_type (str): the relationship type

        Returns:
            dict[str, str]: the relationship properties {name: type} mapping for display
        """
        for relationship_type in schema_object.relationship_types:
            if relationship_type.label != rel_type:
                continue
            return {
                _format_property_name(p): p.type for p in relationship_type.properties
            }
        return {}

    def _node_properties(node_type: NodeType) -> dict[str, str]:
        """Returns a dict {prop_name: prop_type} for all node properties.

        Args:
            node_type (NodeType): the node type object

        Returns:
            dict[str, str]: the node properties {name: type} mapping for display
        """
        return {_format_property_name(p): p.type for p in node_type.properties}

    nodes = [
        Node(  # type: ignore
            id=node_type.label,
            caption=node_type.label,
            properties=_node_properties(node_type),
        )
        for node_type in schema_object.node_types
    ]
    relationships = [
        Relationship(  # type: ignore
            source=pattern[0],
            target=pattern[2],
            caption=pattern[1],
            properties=_relationship_properties(pattern[1]),
        )
        for pattern in schema_object.patterns
    ]

    VG = VisualizationGraph(nodes=nodes, relationships=relationships)
    VG.color_nodes(field="caption")
    return VG
