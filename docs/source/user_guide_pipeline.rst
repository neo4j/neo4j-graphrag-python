.. _user-guide-pipeline:

User Guide: Pipeline
####################

This page provides information about how to create a pipeline.


.. note::

    Pipelines run asynchronously, see examples below.


*******************
Creating Components
*******************

Components are asynchronous units of work that perform simple tasks,
such as chunking documents or saving results to Neo4j.
This package includes a few default components, but developers can create
their own by following these steps:

1. Create a subclass of the Pydantic `neo4j_graphrag.experimental.pipeline.DataModel` to represent the data being returned by the component
2. Create a subclass of `neo4j_graphrag.experimental.pipeline.Component`
3. Create a run method in this new class and specify the required inputs and output model using the just created `DataModel`
4. Implement the run method: it's an `async` method, allowing tasks to be parallelized and awaited within this method.

An example is given below, where a `ComponentAdd` is created to add two numbers together and return
the resulting sum:

.. code:: python

    from neo4j_graphrag.experimental.pipeline import Component, DataModel

    class IntResultModel(DataModel):
        result: int

    class ComponentAdd(Component):
        async def run(self, number1: int, number2: int = 1) -> IntResultModel:
            return IntResultModel(result = number1 + number2)

Read more about :ref:`components-section` in the API Documentation.

***************************************
Connecting Components within a Pipeline
***************************************

The ultimate aim of creating components is to assemble them into a complex pipeline
for a specific purpose, such as building a Knowledge Graph from text data.

Here's how to create a simple pipeline and propagate results from one component to another
(detailed explanations follow):

.. code:: python

    import asyncio
    from neo4j_graphrag.experimental.pipeline import Pipeline

    pipe = Pipeline()
    pipe.add_component(ComponentAdd(), "a")
    pipe.add_component(ComponentAdd(), "b")

    pipe.connect("a", "b", input_config={"number2": "a.result"})
    asyncio.run(pipe.run({"a": {"number1": 10, "number2": 1}, "b": {"number1": 4}}))
    # result: 10+1+4 = 15

1. First, a pipeline is created, and two components named "a" and "b" are added to it.
2. Next, the two components are connected so that "b" runs after "a", with the "number2" parameter for component "b" being the result of component "a".
3. Finally, the pipeline is run with 10 and 1 as input parameters for "a". Component "b" will receive 11 (10 + 1, the result of "a") as "number1" and 4 as "number2" (as specified in the pipeline.run parameters).

The data flow is illustrated in the diagram below:

.. code-block::

    10 ---\
            Component "a" -> 11
    1 ----/                   \
                               \
                                 Component "b" -> 15
    4 -------------------------/

.. warning:: Cyclic graph

    Cycles are not allowed in a Pipeline.


.. warning:: Ignored user inputs

    If inputs are provided both by user in the `pipeline.run` method and as
    `input_config` in a connect method, the user input will be ignored. Take for
    instance the following pipeline, adapted from the previous one:

    .. code:: python

            pipe.connect("a", "b", input_config={"number2": "a.result"})
            asyncio.run(pipe.run({"a": {"number1": 10, "number2": 1}, "b": {"number1": 4, "number2": 42}}))

    The result will still be **15** because the user input `"number2": 42` is ignored.


**********************
Visualising a Pipeline
**********************

Pipelines can be visualized using the `draw` method:

.. code:: python

    import asyncio
    from neo4j_graphrag.experimental.pipeline import Pipeline

    pipe = Pipeline()
    # ... define components and connections

    pipe.draw("pipeline.png")

Here is an example pipeline rendering:

.. image:: images/pipeline_no_unused_outputs.png
  :alt: Pipeline visualisation with hidden outputs if unused


By default, output fields which are not mapped to any component are hidden. They
can be added to the canvas by setting `hide_unused_outputs` to `False`:

.. code:: python

    pipe.draw("pipeline.png", hide_unused_outputs=False)

Here is an example of final result:

.. image:: images/pipeline_full.png
  :alt: Pipeline visualisation
