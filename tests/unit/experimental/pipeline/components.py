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
from neo4j_graphrag.experimental.pipeline import Component, DataModel


class StringResultModel(DataModel):
    result: str


class IntResultModel(DataModel):
    result: int


class ComponentNoParam(Component):
    async def run(self) -> StringResultModel:
        return StringResultModel(result="")


class ComponentPassThrough(Component):
    async def run(self, value: str) -> StringResultModel:
        return StringResultModel(result=f"value is: {value}")


class ComponentAdd(Component):
    async def run(self, number1: int, number2: int) -> IntResultModel:
        return IntResultModel(result=number1 + number2)


class ComponentMultiply(Component):
    async def run(self, number1: int, number2: int = 2) -> IntResultModel:
        return IntResultModel(result=number1 * number2)
