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
from unittest.mock import patch

import neo4j
from neo4j_graphrag.embeddings import Embedder, OpenAIEmbeddings
from neo4j_graphrag.experimental.pipeline.config.object_config import (
    EmbedderConfig,
    EmbedderType,
    LLMConfig,
    LLMType,
    Neo4jDriverConfig,
    Neo4jDriverType,
)
from neo4j_graphrag.llm import LLMInterface, OpenAILLM


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
        "neo4j_graphrag.experimental.pipeline.config.object_config.neo4j.GraphDatabase.driver"
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


def test_llm_config() -> None:
    config = LLMConfig.model_validate(
        {
            "class_": "OpenAILLM",
            "params_": {"model_name": "gpt-4o", "api_key": "my-api-key"},
        }
    )
    assert config.class_ == "OpenAILLM"
    assert config.get_module() == "neo4j_graphrag.llm"
    assert config.get_interface() == LLMInterface
    assert config.params_ == {"model_name": "gpt-4o", "api_key": "my-api-key"}
    d = config.parse()
    assert isinstance(d, OpenAILLM)


def test_llm_type_with_driver(llm: LLMInterface) -> None:
    llm_type = LLMType(llm)
    assert llm_type.parse() == llm


def test_llm_type_with_config() -> None:
    llm_type = LLMType(
        LLMConfig(
            class_="OpenAILLM",
            params_={"model_name": "gpt-4o", "api_key": "my-api-key"},
        )
    )
    llm = llm_type.parse()
    assert isinstance(llm, OpenAILLM)


def test_embedder_config() -> None:
    config = EmbedderConfig.model_validate(
        {
            "class_": "OpenAIEmbeddings",
            "params_": {"api_key": "my-api-key"},
        }
    )
    assert config.class_ == "OpenAIEmbeddings"
    assert config.get_module() == "neo4j_graphrag.embeddings"
    assert config.get_interface() == Embedder
    assert config.params_ == {"api_key": "my-api-key"}
    d = config.parse()
    assert isinstance(d, OpenAIEmbeddings)


def test_embedder_type_with_embedder(embedder: Embedder) -> None:
    embedder_type = EmbedderType(embedder)
    assert embedder_type.parse() == embedder


def test_embedder_type_with_config() -> None:
    embedder_type = EmbedderType(
        EmbedderConfig(
            class_="OpenAIEmbeddings",
            params_={"api_key": "my-api-key"},
        )
    )
    embedder = embedder_type.parse()
    assert isinstance(embedder, OpenAIEmbeddings)
