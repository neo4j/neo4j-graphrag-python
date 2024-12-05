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

from neo4j_graphrag.experimental.pipeline import Pipeline
from neo4j_graphrag.experimental.pipeline.config.pipeline_config import PipelineConfig
from neo4j_graphrag.experimental.pipeline.config.runner import PipelineRunner
from neo4j_graphrag.experimental.pipeline.types import PipelineDefinition


@patch("neo4j_graphrag.experimental.pipeline.pipeline.Pipeline.from_definition")
def test_pipeline_runner_from_def_empty(mock_from_definition: Mock) -> None:
    mock_from_definition.return_value = Pipeline()
    runner = PipelineRunner(
        pipeline_definition=PipelineDefinition(components=[], connections=[])
    )
    assert runner.config is None
    assert runner.pipeline is not None
    assert runner.pipeline._nodes == {}
    assert runner.pipeline._edges == []
    assert runner.run_params == {}
    mock_from_definition.assert_called_once()


def test_pipeline_runner_from_config() -> None:
    config = PipelineConfig(component_config={}, connection_config=[])
    runner = PipelineRunner.from_config(config)
    assert runner.config is not None
    assert runner.pipeline is not None
    assert runner.pipeline._nodes == {}
    assert runner.pipeline._edges == []
    assert runner.run_params == {}


@patch("neo4j_graphrag.experimental.pipeline.config.runner.PipelineRunner.from_config")
@patch("neo4j_graphrag.experimental.pipeline.config.config_reader.ConfigReader.read")
def test_pipeline_runner_from_config_file(
    mock_read: Mock, mock_from_config: Mock
) -> None:
    mock_read.return_value = {"dict": "with data"}
    PipelineRunner.from_config_file("file.yaml")

    mock_read.assert_called_once_with("file.yaml")
    mock_from_config.assert_called_once_with({"dict": "with data"})
