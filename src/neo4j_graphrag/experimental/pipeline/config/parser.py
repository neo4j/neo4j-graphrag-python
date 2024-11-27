import importlib
from typing import Any, Optional, Union

import neo4j

from neo4j_graphrag.experimental.pipeline.config.param_resolvers import PARAM_RESOLVERS
from neo4j_graphrag.experimental.pipeline.config.types import (
    ClassConfig,
    DriverConfig,
    ParamConfig,
    ParamToResolveConfig,
    SimpleKGPipelineConfig,
)
from neo4j_graphrag.experimental.pipeline.kg_builder import (
    SimpleKGPipeline,
    SimpleKGPipelineModel,
)


class SimpleKGPipelineConfigParser:
    RESOLVER_KEY = "resolver_"

    def __init__(self, config: SimpleKGPipelineConfig) -> None:
        self.config = config

    def _get_class(self, klass_path: str, optional_module: Optional[str] = None) -> Any:
        """Get class type from string and an optional module"""
        *modules, klass_name = klass_path.rsplit(".", 1)
        module_name = modules[0] if modules else optional_module
        if module_name is None:
            raise ValueError("Must specify a module to import class from")
        try:
            module = importlib.import_module(module_name)
            klass = getattr(module, klass_name)
        except (ImportError, AttributeError):
            if optional_module and module_name != optional_module:
                full_klass_path = optional_module + "." + klass_path
                return self._get_class(full_klass_path)
            raise ValueError(f"Could not find {klass_name} in {module_name}")
        return klass

    def _parse_neo4j_config(self, neo4j_config: DriverConfig) -> neo4j.Driver:
        driver_init_params = {
            "uri": self._resolve_param(neo4j_config.uri),
            "user": self._resolve_param(neo4j_config.user),
            "password": self._resolve_param(neo4j_config.password),
        }
        # note: we could add a "class" parameter in the config to support async
        # driver in the future. For now, since it's not supported anywhere in
        # the pipeline, we're assuming sync driver is needed
        driver = neo4j.GraphDatabase.driver(
            driver_init_params.pop("uri"),
            auth=(
                driver_init_params.pop("user"),
                driver_init_params.pop("password"),
            ),
            **driver_init_params,
        )
        # Note: driver connectivity checks are delegated to the classed using it
        return driver

    def _get_object(
        self, config: ClassConfig, optional_module: str | None = None
    ) -> Any:
        klass_name = config.class_
        try:
            klass = self._get_class(
                klass_name,
                optional_module=optional_module,
            )
        except ValueError:
            raise ValueError(f"Class '{klass_name}' not found in '{optional_module}'")
        init_params = {}
        for key, param in config.params_.items():
            init_params[key] = self._resolve_param(param)
        embedder = klass(**init_params)
        return embedder

    def _resolve_param(
        self, param: ParamConfig
    ) -> Union[float, str, dict[str, Any], list[Any]]:
        """Recursively resolve parameter"""
        if isinstance(param, list):
            return [self._resolve_param(p) for p in param]
        if not isinstance(param, ParamToResolveConfig):
            return param
        resolver_name = param.resolver_
        resolver_klass = PARAM_RESOLVERS[resolver_name]
        resolver = resolver_klass()
        return resolver.resolve(param)

    def _parse_config(self) -> SimpleKGPipelineModel:
        return SimpleKGPipelineModel(
            driver=self._parse_neo4j_config(self.config.neo4j_config),
            llm=self._get_object(
                self.config.llm_config, optional_module="neo4j_graphrag.llm"
            ),
            embedder=self._get_object(
                self.config.embedder_config, optional_module="neo4j_graphrag.embeddings"
            ),
            from_pdf=self.config.from_pdf,
            entities=list(self.config.entities) if self.config.entities else [],
            relations=list(self.config.relations) if self.config.relations else [],
            potential_schema=list(self.config.potential_schema) if self.config.potential_schema else [],
            pdf_loader=self._get_object(
                self.config.pdf_loader,
                optional_module="neo4j_graphrag.experimental.components.pdf_loader",
            )
            if self.config.pdf_loader
            else None,
            text_splitter=self._get_object(
                self.config.text_splitter,
                optional_module="neo4j_graphrag.experimental.components.text_splitters",
            )
            if self.config.text_splitter
            else None,
            kg_writer=self._get_object(
                self.config.kg_writer,
                optional_module="neo4j_graphrag.experimental.components.kg_writer",
            )
            if self.config.kg_writer
            else None,
            on_error=self.config.on_error,
            prompt_template=self.config.prompt_template,
            perform_entity_resolution=self.config.perform_entity_resolution,
            lexical_graph_config=self.config.lexical_graph_config,
            neo4j_database=self.config.neo4j_database,
        )

    def parse(self) -> SimpleKGPipeline:
        input_model = self._parse_config()
        print(input_model)
        return SimpleKGPipeline(**input_model.model_dump())
