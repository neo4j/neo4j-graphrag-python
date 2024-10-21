"""This examples shows how to create a custom component
that can be added to a Pipeline with:

c = MyComponent(min_value=0, max_value=10)
pipe = Pipeline()
pipe.add_component(c, name="my_component")
"""

import random

from neo4j_graphrag.experimental.pipeline import Component, DataModel
from pydantic import BaseModel, validate_call


class ComponentInputModel(BaseModel):
    """A class to model the component inputs.
    This is not required, inputs can also be passed individually.

    Note: can also inherit from DataModel.
    """

    text: str


class ComponentResultModel(DataModel):
    """A class to model the component outputs.
    Each component must have such a description of the output,
    so that the parameter mapping can be validated before the
    pipeline run starts.
    """

    value: int
    text: str


class MyComponent(Component):
    """Multiplies an input text by a random number
    between `min_value` and `max_value`
    """

    def __init__(self, min_value: int, max_value: int) -> None:
        self.min_value = min_value
        self.max_value = max_value

    # this decorator is required when a Pydantic model is used in the inputs
    @validate_call
    async def run(self, inputs: ComponentInputModel) -> ComponentResultModel:
        # logic here
        random_value = random.randint(self.min_value, self.max_value)
        return ComponentResultModel(
            value=random_value,
            text=inputs.text * random_value,
        )


if __name__ == "__main__":
    import asyncio

    c = MyComponent(min_value=0, max_value=10)
    print(
        asyncio.run(
            c.run(
                inputs={"text": "Hello"}  # type: ignore
            )
        )
    )
