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
3. Create a `run` method in this new class and specify the required inputs and output model using the just created `DataModel`
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

    from neo4j_graphrag.experimental.pipeline import Pipeline

    pipe = Pipeline()
    # ... define components and connections

    pipe.draw("pipeline.html")

Here is an example pipeline rendering as an interactive HTML visualization:

.. code:: python

    # To view the visualization in a browser
    import webbrowser
    webbrowser.open("pipeline.html")

By default, output fields which are not mapped to any component are hidden. They
can be added to the visualization by setting `hide_unused_outputs` to `False`:

.. code:: python

    pipe.draw("pipeline_full.html", hide_unused_outputs=False)
    
    # To view the full visualization in a browser
    import webbrowser
    webbrowser.open("pipeline_full.html")


************************
Adding an Event Callback
************************

It is possible to add a callback to receive notification about pipeline progress:

- `PIPELINE_STARTED`, when pipeline starts
- `PIPELINE_FINISHED`, when pipeline ends
- `TASK_STARTED`, when a task starts
- `TASK_PROGRESS`, sent by each component (depends on component's implementation, see below)
- `TASK_FINISHED`, when a task ends


See :ref:`pipelineevent` and :ref:`taskevent` to see what is sent in each event type.

.. code:: python

    import asyncio
    import logging

    from neo4j_graphrag.experimental.pipeline import Pipeline
    from neo4j_graphrag.experimental.pipeline.notification import Event

    logger = logging.getLogger(__name__)
    logging.basicConfig()
    logger.setLevel(logging.WARNING)


    async def event_handler(event: Event) -> None:
        """Function can do anything about the event,
        here we're just logging it if it's a pipeline-level event.
        """
        if event.event_type.is_pipeline_event:
            logger.warning(event)

    pipeline = Pipeline(
        callback=event_handler,
    )
    # ... add components, connect them as usual

    await pipeline.run(...)


Send Events from Components
===========================

Components can send progress notifications using the `notify` function from
`context_` by implementing the `run_from_context` method:

.. code:: python

    from neo4j_graphrag.experimental.pipeline import Component, DataModel
    from neo4j_graphrag.experimental.pipeline.types.context import RunContext

    class IntResultModel(DataModel):
        result: int

    class ComponentAdd(Component):
        async def run_with_context(self, context_: RunContext, number1: int, number2: int = 1) -> IntResultModel:
            for fake_iteration in range(10):
                await context_.notify(
                    message=f"Starting iteration {fake_iteration} out of 10",
                    data={"iteration": fake_iteration, "total": 10}
                )
            return IntResultModel(result = number1 + number2)

This will send an `TASK_PROGRESS` event to the pipeline callback.

.. note::

    In a future release, the `context_` parameter will be added to the `run` method.
