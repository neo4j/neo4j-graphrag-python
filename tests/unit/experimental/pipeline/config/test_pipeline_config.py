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
from unittest.mock import Mock, patch

import neo4j
from neo4j_graphrag.embeddings import Embedder
from neo4j_graphrag.experimental.pipeline import Component
from neo4j_graphrag.experimental.pipeline.config.object_config import (
    ComponentConfig,
    ComponentType,
    Neo4jDriverConfig,
    Neo4jDriverType,
)
from neo4j_graphrag.experimental.pipeline.config.param_resolver import (
    ParamFromEnvConfig,
    ParamFromKeyConfig,
)
from neo4j_graphrag.experimental.pipeline.config.pipeline_config import (
    AbstractPipelineConfig,
)
from neo4j_graphrag.experimental.pipeline.types.definitions import ComponentDefinition
from neo4j_graphrag.llm import LLMInterface


@patch(
    "neo4j_graphrag.experimental.pipeline.config.object_config.Neo4jDriverConfig.parse"
)
def test_abstract_pipeline_config_neo4j_config_is_a_dict_with_params_(
    mock_neo4j_config: Mock,
) -> None:
    mock_neo4j_config.return_value = "text"
    config = AbstractPipelineConfig.model_validate(
        {
            "neo4j_config": {
                "params_": {
                    "uri": "bolt://",
                    "user": "",
                    "password": "",
                }
            }
        }
    )
    assert isinstance(config.neo4j_config, dict)
    assert "default" in config.neo4j_config
    config.parse()
    mock_neo4j_config.assert_called_once()
    assert config._global_data["neo4j_config"]["default"] == "text"


@patch(
    "neo4j_graphrag.experimental.pipeline.config.object_config.Neo4jDriverConfig.parse"
)
def test_abstract_pipeline_config_neo4j_config_is_a_dict_with_names(
    mock_neo4j_config: Mock,
) -> None:
    mock_neo4j_config.return_value = "text"
    config = AbstractPipelineConfig.model_validate(
        {
            "neo4j_config": {
                "my_driver": {
                    "params_": {
                        "uri": "bolt://",
                        "user": "",
                        "password": "",
                    }
                }
            }
        }
    )
    assert isinstance(config.neo4j_config, dict)
    assert "my_driver" in config.neo4j_config
    config.parse()
    mock_neo4j_config.assert_called_once()
    assert config._global_data["neo4j_config"]["my_driver"] == "text"


@patch(
    "neo4j_graphrag.experimental.pipeline.config.object_config.Neo4jDriverConfig.parse"
)
def test_abstract_pipeline_config_neo4j_config_is_a_dict_with_driver(
    mock_neo4j_config: Mock, driver: neo4j.Driver
) -> None:
    config = AbstractPipelineConfig.model_validate(
        {
            "neo4j_config": {
                "my_driver": driver,
            }
        }
    )
    assert isinstance(config.neo4j_config, dict)
    assert "my_driver" in config.neo4j_config
    config.parse()
    assert not mock_neo4j_config.called
    assert config._global_data["neo4j_config"]["my_driver"] == driver


@patch(
    "neo4j_graphrag.experimental.pipeline.config.object_config.Neo4jDriverConfig.parse"
)
def test_abstract_pipeline_config_neo4j_config_is_a_driver(
    mock_neo4j_config: Mock, driver: neo4j.Driver
) -> None:
    config = AbstractPipelineConfig.model_validate(
        {
            "neo4j_config": driver,
        }
    )
    assert isinstance(config.neo4j_config, dict)
    assert "default" in config.neo4j_config
    config.parse()
    assert not mock_neo4j_config.called
    assert config._global_data["neo4j_config"]["default"] == driver


@patch("neo4j_graphrag.experimental.pipeline.config.object_config.LLMConfig.parse")
def test_abstract_pipeline_config_llm_config_is_a_dict_with_params_(
    mock_llm_config: Mock,
) -> None:
    mock_llm_config.return_value = "text"
    config = AbstractPipelineConfig.model_validate(
        {"llm_config": {"class_": "OpenAILLM", "params_": {"model_name": "gpt-4o"}}}
    )
    assert isinstance(config.llm_config, dict)
    assert "default" in config.llm_config
    config.parse()
    mock_llm_config.assert_called_once()
    assert config._global_data["llm_config"]["default"] == "text"


@patch("neo4j_graphrag.experimental.pipeline.config.object_config.LLMConfig.parse")
def test_abstract_pipeline_config_llm_config_is_a_dict_with_names(
    mock_llm_config: Mock,
) -> None:
    mock_llm_config.return_value = "text"
    config = AbstractPipelineConfig.model_validate(
        {
            "llm_config": {
                "my_llm": {"class_": "OpenAILLM", "params_": {"model_name": "gpt-4o"}}
            }
        }
    )
    assert isinstance(config.llm_config, dict)
    assert "my_llm" in config.llm_config
    config.parse()
    mock_llm_config.assert_called_once()
    assert config._global_data["llm_config"]["my_llm"] == "text"


@patch("neo4j_graphrag.experimental.pipeline.config.object_config.LLMConfig.parse")
def test_abstract_pipeline_config_llm_config_is_a_dict_with_llm(
    mock_llm_config: Mock, llm: LLMInterface
) -> None:
    config = AbstractPipelineConfig.model_validate(
        {
            "llm_config": {
                "my_llm": llm,
            }
        }
    )
    assert isinstance(config.llm_config, dict)
    assert "my_llm" in config.llm_config
    config.parse()
    assert not mock_llm_config.called
    assert config._global_data["llm_config"]["my_llm"] == llm


@patch("neo4j_graphrag.experimental.pipeline.config.object_config.LLMConfig.parse")
def test_abstract_pipeline_config_llm_config_is_a_llm(
    mock_llm_config: Mock, llm: LLMInterface
) -> None:
    config = AbstractPipelineConfig.model_validate(
        {
            "llm_config": llm,
        }
    )
    assert isinstance(config.llm_config, dict)
    assert "default" in config.llm_config
    config.parse()
    assert not mock_llm_config.called
    assert config._global_data["llm_config"]["default"] == llm


@patch("neo4j_graphrag.experimental.pipeline.config.object_config.EmbedderConfig.parse")
def test_abstract_pipeline_config_embedder_config_is_a_dict_with_params_(
    mock_embedder_config: Mock,
) -> None:
    mock_embedder_config.return_value = "text"
    config = AbstractPipelineConfig.model_validate(
        {"embedder_config": {"class_": "OpenAIEmbeddings", "params_": {}}}
    )
    assert isinstance(config.embedder_config, dict)
    assert "default" in config.embedder_config
    config.parse()
    mock_embedder_config.assert_called_once()
    assert config._global_data["embedder_config"]["default"] == "text"


@patch("neo4j_graphrag.experimental.pipeline.config.object_config.EmbedderConfig.parse")
def test_abstract_pipeline_config_embedder_config_is_a_dict_with_names(
    mock_embedder_config: Mock,
) -> None:
    mock_embedder_config.return_value = "text"
    config = AbstractPipelineConfig.model_validate(
        {
            "embedder_config": {
                "my_embedder": {"class_": "OpenAIEmbeddings", "params_": {}}
            }
        }
    )
    assert isinstance(config.embedder_config, dict)
    assert "my_embedder" in config.embedder_config
    config.parse()
    mock_embedder_config.assert_called_once()
    assert config._global_data["embedder_config"]["my_embedder"] == "text"


@patch("neo4j_graphrag.experimental.pipeline.config.object_config.EmbedderConfig.parse")
def test_abstract_pipeline_config_embedder_config_is_a_dict_with_llm(
    mock_embedder_config: Mock, embedder: Embedder
) -> None:
    config = AbstractPipelineConfig.model_validate(
        {
            "embedder_config": {
                "my_embedder": embedder,
            }
        }
    )
    assert isinstance(config.embedder_config, dict)
    assert "my_embedder" in config.embedder_config
    config.parse()
    assert not mock_embedder_config.called
    assert config._global_data["embedder_config"]["my_embedder"] == embedder


@patch("neo4j_graphrag.experimental.pipeline.config.object_config.EmbedderConfig.parse")
def test_abstract_pipeline_config_embedder_config_is_an_embedder(
    mock_embedder_config: Mock, embedder: Embedder
) -> None:
    config = AbstractPipelineConfig.model_validate(
        {
            "embedder_config": embedder,
        }
    )
    assert isinstance(config.embedder_config, dict)
    assert "default" in config.embedder_config
    config.parse()
    assert not mock_embedder_config.called
    assert config._global_data["embedder_config"]["default"] == embedder


def test_abstract_pipeline_config_parse_global_data_no_extras(driver: Mock) -> None:
    config = AbstractPipelineConfig(
        neo4j_config={"my_driver": Neo4jDriverType(driver)},
    )
    gd = config._parse_global_data()
    assert gd == {
        "extras": {},
        "neo4j_config": {
            "my_driver": driver,
        },
        "llm_config": {},
        "embedder_config": {},
    }


@patch(
    "neo4j_graphrag.experimental.pipeline.config.param_resolver.ParamFromEnvConfig.resolve"
)
def test_abstract_pipeline_config_parse_global_data_extras(
    mock_param_resolver: Mock,
) -> None:
    mock_param_resolver.return_value = "my value"
    config = AbstractPipelineConfig(
        extras={"my_extra_var": ParamFromEnvConfig(var_="some key")},
    )
    gd = config._parse_global_data()
    assert gd == {
        "extras": {"my_extra_var": "my value"},
        "neo4j_config": {},
        "llm_config": {},
        "embedder_config": {},
    }


@patch(
    "neo4j_graphrag.experimental.pipeline.config.param_resolver.ParamFromEnvConfig.resolve"
)
@patch(
    "neo4j_graphrag.experimental.pipeline.config.object_config.Neo4jDriverType.parse"
)
def test_abstract_pipeline_config_parse_global_data_use_extras_in_other_config(
    mock_neo4j_parser: Mock,
    mock_param_resolver: Mock,
) -> None:
    """Parser is able to read variables in the 'extras' section of config
    to instantiate another object (neo4j.Driver in this test case)
    """
    mock_param_resolver.side_effect = ["bolt://myhost", "myuser", "mypwd"]
    mock_neo4j_parser.return_value = "my driver"
    config = AbstractPipelineConfig(
        extras={
            "my_extra_uri": ParamFromEnvConfig(var_="some key"),
            "my_extra_user": ParamFromEnvConfig(var_="some key"),
            "my_extra_pwd": ParamFromEnvConfig(var_="some key"),
        },
        neo4j_config={
            "my_driver": Neo4jDriverType(
                Neo4jDriverConfig(
                    params_=dict(
                        uri=ParamFromKeyConfig(key_="extras.my_extra_uri"),
                        user=ParamFromKeyConfig(key_="extras.my_extra_user"),
                        password=ParamFromKeyConfig(key_="extras.my_extra_pwd"),
                    )
                )
            )
        },
    )
    gd = config._parse_global_data()
    expected_extras = {
        "my_extra_uri": "bolt://myhost",
        "my_extra_user": "myuser",
        "my_extra_pwd": "mypwd",
    }
    assert gd["extras"] == expected_extras
    assert gd["neo4j_config"] == {"my_driver": "my driver"}
    mock_neo4j_parser.assert_called_once_with({"extras": expected_extras})


@patch("neo4j_graphrag.experimental.pipeline.config.object_config.ComponentType.parse")
def test_abstract_pipeline_config_resolve_component_definition_no_run_params(
    mock_component_parse: Mock,
    component: Component,
) -> None:
    mock_component_parse.return_value = component
    config = AbstractPipelineConfig()
    component_type = ComponentType(component)
    component_definition = config._resolve_component_definition("name", component_type)
    assert isinstance(component_definition, ComponentDefinition)
    mock_component_parse.assert_called_once_with({})
    assert component_definition.name == "name"
    assert component_definition.component == component
    assert component_definition.run_params == {}


@patch(
    "neo4j_graphrag.experimental.pipeline.config.pipeline_config.AbstractPipelineConfig.resolve_params"
)
@patch("neo4j_graphrag.experimental.pipeline.config.object_config.ComponentType.parse")
def test_abstract_pipeline_config_resolve_component_definition_with_run_params(
    mock_component_parse: Mock,
    mock_resolve_params: Mock,
    component: Component,
) -> None:
    mock_component_parse.return_value = component
    mock_resolve_params.return_value = {"param": "resolver param result"}
    config = AbstractPipelineConfig()
    component_type: ComponentType[Component] = ComponentType(
        ComponentConfig(class_="", params_={}, run_params_={"param1": "value1"})
    )
    component_definition = config._resolve_component_definition("name", component_type)
    assert isinstance(component_definition, ComponentDefinition)
    mock_component_parse.assert_called_once_with({})
    assert component_definition.name == "name"
    assert component_definition.component == component
    assert component_definition.run_params == {"param": "resolver param result"}
    mock_resolve_params.assert_called_once_with({"param1": "value1"})
