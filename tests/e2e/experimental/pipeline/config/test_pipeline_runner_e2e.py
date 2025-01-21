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
import os
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
from neo4j import Driver
from neo4j_graphrag.experimental.pipeline.config.runner import PipelineRunner
from neo4j_graphrag.experimental.pipeline.pipeline import PipelineResult
from neo4j_graphrag.llm import LLMResponse


@pytest.fixture(scope="function", autouse=True)
def clear_db(driver: Driver) -> Any:
    driver.execute_query("MATCH (n) DETACH DELETE n")
    yield


@pytest.mark.asyncio
async def test_pipeline_from_json_config(harry_potter_text: str, driver: Mock) -> None:
    os.environ["NEO4J_URI"] = "neo4j://localhost:7687"
    os.environ["NEO4J_USER"] = "neo4j"
    os.environ["NEO4J_PASSWORD"] = "password"

    runner = PipelineRunner.from_config_file(
        "tests/e2e/data/config_files/pipeline_config.json"
    )
    res = await runner.run({"splitter": {"text": harry_potter_text}})
    assert isinstance(res, PipelineResult)
    assert res.result["writer"]["metadata"] == {
        "node_count": 11,
        "relationship_count": 10,
    }
    nodes = driver.execute_query("MATCH (n) RETURN n")
    assert len(nodes.records) == 11


@pytest.mark.asyncio
async def test_pipeline_from_yaml_config(harry_potter_text: str, driver: Mock) -> None:
    os.environ["NEO4J_URI"] = "neo4j://localhost:7687"
    os.environ["NEO4J_USER"] = "neo4j"
    os.environ["NEO4J_PASSWORD"] = "password"

    runner = PipelineRunner.from_config_file(
        "tests/e2e/data/config_files/pipeline_config.yaml"
    )
    res = await runner.run({"splitter": {"text": harry_potter_text}})
    assert isinstance(res, PipelineResult)
    assert res.result["writer"]["metadata"] == {
        "node_count": 11,
        "relationship_count": 10,
    }

    nodes = driver.execute_query("MATCH (n) RETURN n")
    assert len(nodes.records) == 11


@patch(
    "neo4j_graphrag.experimental.pipeline.config.runner.SimpleKGPipelineConfig.get_default_embedder"
)
@patch(
    "neo4j_graphrag.experimental.pipeline.config.runner.SimpleKGPipelineConfig.get_default_llm"
)
@pytest.mark.asyncio
async def test_simple_kg_pipeline_from_json_config(
    mock_llm: Mock, mock_embedder: Mock, harry_potter_text: str, driver: Mock
) -> None:
    mock_llm.return_value.ainvoke = AsyncMock(
        side_effect=[
            LLMResponse(
                content="""{
                            "nodes": [
                                {
                                    "id": "0",
                                    "label": "Person",
                                    "properties": {
                                        "name": "Harry Potter"
                                    }
                                },
                                {
                                    "id": "1",
                                    "label": "Person",
                                    "properties": {
                                        "name": "Alastor Mad-Eye Moody"
                                    }
                                },
                                {
                                    "id": "2",
                                    "label": "Organization",
                                    "properties": {
                                        "name": "The Order of the Phoenix"
                                    }
                                }
                            ],
                            "relationships": [
                                {
                                    "type": "KNOWS",
                                    "start_node_id": "0",
                                    "end_node_id": "1"
                                },
                                {
                                    "type": "LED_BY",
                                    "start_node_id": "2",
                                    "end_node_id": "1"
                                }
                            ]
                        }"""
            ),
        ]
    )
    mock_embedder.return_value.embed_query.side_effect = [
        [1.0, 2.0],
    ]

    os.environ["NEO4J_URI"] = "neo4j://localhost:7687"
    os.environ["NEO4J_USER"] = "neo4j"
    os.environ["NEO4J_PASSWORD"] = "password"
    os.environ["OPENAI_API_KEY"] = "sk-my-secret-key"
    os.environ["MY_OPENAI_KEY"] = "my-openai-key"

    runner = PipelineRunner.from_config_file(
        "tests/e2e/data/config_files/simple_kg_pipeline_config.json"
    )

    # check extras and API keys are handled as expected
    config = runner.config
    assert config is not None
    # extras must be resolved:
    assert config._global_data["extras"] == {"openai_api_key": "my-openai-key"}
    # API key for LLM is read from "extras" (see config file)
    default_llm = config._global_data["llm_config"]["default"]
    assert default_llm.client.api_key == "my-openai-key"
    # API key for embedder is read from env vars (see config file)
    default_embedder = config._global_data["embedder_config"]["default"]
    assert default_embedder.client.api_key == "sk-my-secret-key"  # read from env vaf

    # then run pipeline and check results
    res = await runner.run({"file_path": "tests/e2e/data/documents/harry_potter.pdf"})
    assert isinstance(res, PipelineResult)
    assert res.result["resolver"] == {
        "number_of_nodes_to_resolve": 3,
        "number_of_created_nodes": 3,
    }
    nodes = driver.execute_query("MATCH (n) RETURN n")
    # 1 chunk + 1 document + 3 __Entity__ nodes
    assert len(nodes.records) == 5


@patch(
    "neo4j_graphrag.experimental.pipeline.config.runner.SimpleKGPipelineConfig.get_default_embedder"
)
@patch(
    "neo4j_graphrag.experimental.pipeline.config.runner.SimpleKGPipelineConfig.get_default_llm"
)
@pytest.mark.asyncio
async def test_simple_kg_pipeline_from_yaml_config(
    mock_llm: Mock, mock_embedder: Mock, harry_potter_text: str, driver: Mock
) -> None:
    mock_llm.return_value.ainvoke = AsyncMock(
        side_effect=[
            LLMResponse(
                content="""{
                            "nodes": [
                                {
                                    "id": "0",
                                    "label": "Person",
                                    "properties": {
                                        "name": "Harry Potter"
                                    }
                                },
                                {
                                    "id": "1",
                                    "label": "Person",
                                    "properties": {
                                        "name": "Alastor Mad-Eye Moody"
                                    }
                                },
                                {
                                    "id": "2",
                                    "label": "Organization",
                                    "properties": {
                                        "name": "The Order of the Phoenix"
                                    }
                                }
                            ],
                            "relationships": [
                                {
                                    "type": "KNOWS",
                                    "start_node_id": "0",
                                    "end_node_id": "1"
                                },
                                {
                                    "type": "LED_BY",
                                    "start_node_id": "2",
                                    "end_node_id": "1"
                                }
                            ]
                        }"""
            ),
        ]
    )
    mock_embedder.return_value.embed_query.side_effect = [
        [1.0, 2.0],
    ]

    os.environ["NEO4J_URI"] = "neo4j://localhost:7687"
    os.environ["NEO4J_USER"] = "neo4j"
    os.environ["NEO4J_PASSWORD"] = "password"
    os.environ["OPENAI_API_KEY"] = "sk-my-secret-key"

    runner = PipelineRunner.from_config_file(
        "tests/e2e/data/config_files/simple_kg_pipeline_config.yaml"
    )
    res = await runner.run({"file_path": "tests/e2e/data/documents/harry_potter.pdf"})
    assert isinstance(res, PipelineResult)
    # print(await runner.pipeline.store.get_result_for_component(res.run_id, "splitter"))
    assert res.result["resolver"] == {
        "number_of_nodes_to_resolve": 3,
        "number_of_created_nodes": 3,
    }
    nodes = driver.execute_query("MATCH (n) RETURN n")
    # 1 chunk + 1 document + 3 nodes
    assert len(nodes.records) == 5
