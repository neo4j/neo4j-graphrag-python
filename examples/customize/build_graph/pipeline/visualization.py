#  Copyright (c) "Neo4j"
#  Neo4j Sweden AB [https://neo4j.com]
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#      https://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""This example illustrates how to visualize a Pipeline"""

from neo4j_graphrag.experimental.pipeline import Component, Pipeline
from neo4j_graphrag.experimental.pipeline.component import DataModel
from pydantic import validate_call


class IntDataModel(DataModel):
    value: int
    message: str


class Addition(Component):
    async def run(self, a: int, b: int) -> IntDataModel:
        return IntDataModel(value=a + b, message="addition complete")


class Duplicate(Component):
    def __init__(self, factor: int = 2) -> None:
        self.factor = factor

    async def run(self, number: int) -> IntDataModel:
        return IntDataModel(
            value=number * self.factor, message=f"multiplication by {self.factor} done"
        )


class Save(Component):
    @validate_call
    async def run(self, number: IntDataModel) -> IntDataModel:
        return IntDataModel(value=number.value, message="saved")


if __name__ == "__main__":
    pipe = Pipeline()
    pipe.add_component(Duplicate(), "times_two")
    pipe.add_component(Duplicate(factor=10), "times_ten")
    pipe.add_component(Addition(), "addition")
    pipe.add_component(Save(), "save")
    pipe.connect("times_two", "addition", {"a": "times_two.value"})
    pipe.connect("times_ten", "addition", {"b": "times_ten.value"})
    pipe.connect("addition", "save", {"number": "addition"})
    pipe.draw("graph.png")
    pipe.draw("graph_full.png", hide_unused_outputs=False)
