import pytest

from neo4j_graphrag.tools.tool import Tool, ObjectParameter, StringParameter


class TestTool(Tool):
    """Test tool for unit tests."""

    def __init__(self, name: str = "test_tool", description: str = "A test tool"):
        parameters = ObjectParameter(
            description="Test parameters",
            properties={"param1": StringParameter(description="Test parameter")},
            required_properties=["param1"],
            additional_properties=False,
        )

        super().__init__(
            name=name,
            description=description,
            parameters=parameters,
            execute_func=lambda **kwargs: kwargs,
        )


@pytest.fixture
def test_tool() -> Tool:
    return TestTool()
