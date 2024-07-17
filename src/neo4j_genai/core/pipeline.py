import abc
import asyncio
from typing import Optional, Any


# class ComponentMeta(abc.ABCMeta):
#     def __new__(
#         meta, name: str, bases: tuple[type, ...], attrs: dict[str, Any]
#     ) -> type:
#         run_method = attrs.get("run")
#         if run_method is not None:
#             sig = inspect.signature(run_method)
#             inputs = [k for k in sig.parameters if k != "self"]
#             if "kwargs" in inputs and name != "Component":
#                 raise Exception()
#             attrs["__inputs"] = inputs
#         return type.__new__(meta, name, bases, attrs)


class Component(abc.ABC):  # , metaclass=ComponentMeta):

    @abc.abstractmethod
    def run(self, **kwargs) -> dict:
        pass

    async def arun(self, **kwargs) -> dict:
        """Runs asynchronously
        """
        return self.run(**kwargs)


class Pipeline:
    def __init__(self):
        self._components = {}
        # storing the results for each component,
        # not the best memory-wise?
        # results dict has key 'component_name' and the value is the component output
        self._results = {}

    def add_component(self, component: Component, name: str = "",
                      input_def: Optional[dict[str, Any]] = None) -> None:
        """
        Add a component to the pipeline.

        Args:
            component:
            name: name of the component within the pipeline. Must be unique.
            input_def (dict): mapping between component's inputs and the output of
             previous components in the pipeline
        """
        if name in self._components:
            raise KeyError(f"Component with name '{name}' already in this Pipeline")
        self._components[name] = (component, input_def)

    def _get_component_inputs(self, component_name: str, input_defs: dict, data: dict) -> dict:
        # TODO: use some defaults when the previous component's output has same
        #   name/type (?) as this component's inputs
        component_inputs = data.get(component_name, {})
        if input_defs:
            for input_def, mapping in input_defs.items():
                input_component, param = mapping.split(".")
                value = self._results[input_component][param]
                component_inputs[input_def] = value
        return component_inputs

    def run(self, data: dict):
        if self._results:
            self._results = {}
        r = None
        for name, (component, input_defs) in self._components.items():
            component_inputs = self._get_component_inputs(name, input_defs, data)
            r = component.run(**component_inputs)
            self._results[name] = r
        return r

    async def arun(self, data: dict):
        """
        Calls the arun method for each component.
        """
        if self._results:
            self._results = {}
        r = None
        for name, (component, input_defs) in self._components.items():
            component_inputs = self._get_component_inputs(name, input_defs, data)
            r = await component.arun(**component_inputs)
            self._results[name] = r
        return r


class SentenceSplitter(Component):

    def __init__(self, separator: str = "."):
        super().__init__()
        self.separator = separator

    def run(self, text: str = "") -> dict:
        return {"sentences": [t.strip() for t in text.split(self.separator) if t.strip()]}


class WordCounter(Component):

    def __init__(self, word_delimiter: str = " "):
        super().__init__()
        self.word_delimiter = word_delimiter

    def run(self, texts: list[str], remove_chars: list[str]) -> dict:
        counts = []
        for text in texts:
            for r in remove_chars:
                text = text.replace(r, "")
            counts.append(len([t.strip() for t in text.split(self.word_delimiter) if t.strip()]))
        return {"counts": counts}

    async def _text_counter(self, text, remove_chars):
        for r in remove_chars:
            text = text.replace(r, "")
        return len([t.strip() for t in text.split(self.word_delimiter) if t.strip()])

    async def arun(self, texts: list[str], remove_chars: list[str]) -> dict:
        results = await asyncio.gather(
            *[
                self._text_counter(text, remove_chars)
                for text in texts
            ]
        )
        return {"counts": results}


class WordCounterSimple(Component):

    def __init__(self, word_delimiter: str = " "):
        self.word_delimiter = word_delimiter

    def run(self, text: str, remove_chars: list[str]) -> dict:
        # here we are processing text one by one, how can the pipeline handle that?
        for r in remove_chars:
            text = text.replace(r, "")
        return {"count": len(text.split(self.word_delimiter))}


if __name__ == '__main__':
    pipe = Pipeline()
    pipe.add_component(SentenceSplitter(), "splitter")
    pipe.add_component(WordCounter(), "counter", input_def={
        "texts": "splitter.sentences"
    })

    pipe_inputs = {
        "splitter":
            {
                "text": "Graphs are everywhere. "
                        "GraphRAG is the future of Artificial Intelligence. "
                        "Robots are already running the world."
            },
        "counter": {
            "remove_chars": [",", "is"]
        }
    }
    print(pipe.run(pipe_inputs))
    print(asyncio.run(pipe.arun(pipe_inputs)))
