from .components import ComponentMultiply


def test_component_inputs() -> None:
    inputs = ComponentMultiply.component_inputs  # type: ignore
    assert "number1" in inputs
    assert inputs["number1"]["has_default"] is False
    assert "number2" in inputs
    assert inputs["number2"]["has_default"] is True


def test_component_outputs() -> None:
    outputs = ComponentMultiply.component_outputs  # type: ignore
    assert "result" in outputs
