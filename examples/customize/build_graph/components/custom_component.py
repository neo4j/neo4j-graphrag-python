"""This examples shows how to create a custom component
that can be added to a Pipeline with:

c = MyComponent(min_value=0, max_value=10)
pipe = Pipeline()
pipe.add_component(c, name="my_component")
"""

import random

from neo4j_graphrag.experimental.pipeline import Component, DataModel


class ComponentResultModel(DataModel):
    value: int
    text: str


class MyComponent(Component):
    """Multiplies an input text by a random number
    between `min_value` and `max_value`
    """

    def __init__(self, min_value: int, max_value: int) -> None:
        self.min_value = min_value
        self.max_value = max_value

    async def run(self, input_text: str) -> ComponentResultModel:
        # logic here
        random_value = random.randint(self.min_value, self.max_value)
        return ComponentResultModel(
            value=random_value,
            text=input_text * random_value,
        )


if __name__ == "__main__":
    import asyncio

    c = MyComponent(min_value=0, max_value=10)
    print(asyncio.run(c.run(input_text="Hello")))
