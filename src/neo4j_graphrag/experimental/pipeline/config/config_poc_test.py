from unittest.mock import MagicMock, Mock, patch

import neo4j
import pytest

from neo4j_graphrag.experimental.pipeline.config.config_parser import (
    AbstractPipelineConfig,
    LLMConfig,
    LLMType,
    Neo4jDriverConfig,
    Neo4jDriverType,
)
from neo4j_graphrag.llm import LLMInterface, OpenAILLM


@pytest.fixture(scope="function")
def driver() -> MagicMock:
    return MagicMock(spec=neo4j.Driver)


def test_neo4j_driver_config() -> None:
    config = Neo4jDriverConfig.model_validate(
        {
            "params_": {
                "uri": "bolt://",
                "user": "a user",
                "password": "a password",
            }
        }
    )
    assert config.class_ == "not used"
    assert config.params_ == {
        "uri": "bolt://",
        "user": "a user",
        "password": "a password",
    }
    with patch(
        "neo4j_graphrag.experimental.pipeline.config.config_poc.neo4j.GraphDatabase.driver"
    ) as driver_mock:
        driver_mock.return_value = "a driver"
        d = config.parse()
        driver_mock.assert_called_once_with("bolt://", auth=("a user", "a password"))
        assert d == "a driver"  # type: ignore


def test_neo4j_driver_type_with_driver(driver: neo4j.Driver) -> None:
    driver_type = Neo4jDriverType(driver)
    assert driver_type.parse() == driver


def test_neo4j_driver_type_with_config() -> None:
    driver_type = Neo4jDriverType(
        Neo4jDriverConfig(
            params_={
                "uri": "bolt://",
                "user": "",
                "password": "",
            }
        )
    )
    driver = driver_type.parse()
    assert isinstance(driver, neo4j.Driver)


@patch.multiple(AbstractPipelineConfig, __abstractmethods__=set())
@patch("neo4j_graphrag.experimental.pipeline.config.config_poc.Neo4jDriverConfig.parse")
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


@patch.multiple(AbstractPipelineConfig, __abstractmethods__=set())
@patch("neo4j_graphrag.experimental.pipeline.config.config_poc.Neo4jDriverConfig.parse")
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


@patch.multiple(AbstractPipelineConfig, __abstractmethods__=set())
@patch("neo4j_graphrag.experimental.pipeline.config.config_poc.Neo4jDriverConfig.parse")
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


@patch.multiple(AbstractPipelineConfig, __abstractmethods__=set())
@patch("neo4j_graphrag.experimental.pipeline.config.config_poc.Neo4jDriverConfig.parse")
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


@pytest.fixture(scope="function")
def llm() -> LLMInterface:
    return MagicMock(spec=LLMInterface)


def test_llm_config() -> None:
    config = LLMConfig.model_validate(
        {"class_": "OpenAILLM", "params_": {"model_name": "gpt-4o"}}
    )
    assert config.class_ == "OpenAILLM"
    assert config.get_module() == "neo4j_graphrag.llm"
    assert config.get_interface() == LLMInterface
    assert config.params_ == {"model_name": "gpt-4o"}
    d = config.parse()
    assert isinstance(d, OpenAILLM)


def test_llm_type_with_driver(llm: LLMInterface) -> None:
    llm_type = LLMType(llm)
    assert llm_type.parse() == llm


def test_llm_type_with_config() -> None:
    llm_type = LLMType(LLMConfig(class_="OpenAILLM", params_={"model_name": "gpt-4o"}))
    llm = llm_type.parse()
    assert isinstance(llm, LLMInterface)


@patch.multiple(AbstractPipelineConfig, __abstractmethods__=set())
@patch("neo4j_graphrag.experimental.pipeline.config.config_poc.LLMConfig.parse")
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


@patch.multiple(AbstractPipelineConfig, __abstractmethods__=set())
@patch("neo4j_graphrag.experimental.pipeline.config.config_poc.LLMConfig.parse")
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


@patch.multiple(AbstractPipelineConfig, __abstractmethods__=set())
@patch("neo4j_graphrag.experimental.pipeline.config.config_poc.LLMConfig.parse")
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


@patch.multiple(AbstractPipelineConfig, __abstractmethods__=set())
@patch("neo4j_graphrag.experimental.pipeline.config.config_poc.LLMConfig.parse")
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
