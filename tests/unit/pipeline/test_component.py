from .components import ComponentMultiply


def test_component_inputs():
    inputs = ComponentMultiply.component_inputs
    assert "number1" in inputs
    assert inputs["number1"]["has_default"] is False
    assert "number2" in inputs
    assert inputs["number2"]["has_default"] is True


def test_component_outputs():
    outputs = ComponentMultiply.component_outputs
    assert "result" in outputs
