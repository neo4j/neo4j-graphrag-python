.. _user-guide-pipeline:

User Guide: Pipeline
####################

This page provides information about how to create a pipeline.


******************************
Creating Components
******************************

Components are units of work that perform simple tasks, such as chunking documents
or saving results to Neo4j. This package includes a few default components,
but developers can create their own by following these steps:

1. Create a subclass of `neo4j_genai.pipeline.DataModel` to represent the data being returned by the component
2. Create a subclass of `neo4j_genai.pipeline.Component`
3. Create a run method in this new class and specify the required inputs and output model using the just created `DataModel`
4. Implement the run method

An example is given below:

.. code:: python

    from neo4j_genai.pipeline import Component, DataModel

    class IntResultModel(DataModel):
        result: int

    class ComponentAdd(Component):
        async def run(self, number1: int, number2: int = 1) -> IntResultModel:
            return IntResultModel(result = number1 + number2)


***************************************
Connecting Components within a Pipeline
***************************************

The ultimate aim of creating components is to assemble them into a complex pipeline
for a specific purpose, such as building a Knowledge Graph from text data.

Here's how to create a simple pipeline (detailed explanations follow):

.. code:: python

    import asyncio
    from neo4j_genai.pipeline import Pipeline

    pipe = Pipeline()
    pipe.add_component("a", ComponentAdd())
    pipe.add_component("b", ComponentAdd())

    pipe.connect("a", "b", {"number2": "a.result"})
    asyncio.run(pipe.run({"a": {"number1": 10, "number2": 1}, "b": {"number1": 4}))
    # result: 10+1+4 = 15

1. First, a pipeline is created, and two components named "a" and "b" are added to it.
2. Next, the two components are connected so that "b" runs after "a", with the "number2" parameter for component "b" being the result of component "a".
3. Finally, the pipeline is run with 10 and 1 as input parameters for "a". Component "b" will receive 11 (10 + 1, the result of "a") as "number1" and 4 as "number2" (as specified in the pipeline.run parameters).

.. warning::

    Cycles are not allowed in a Pipeline.
