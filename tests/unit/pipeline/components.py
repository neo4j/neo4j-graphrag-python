from neo4j_genai.pipeline import Component, DataModel


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
