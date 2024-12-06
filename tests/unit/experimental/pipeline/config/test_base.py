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

from neo4j_graphrag.experimental.pipeline.config.base import AbstractConfig
from neo4j_graphrag.experimental.pipeline.config.param_resolver import (
    ParamToResolveConfig,
)


def test_resolve_param_with_param_to_resolve_object() -> None:
    c = AbstractConfig()
    with patch(
        "neo4j_graphrag.experimental.pipeline.config.param_resolver.ParamToResolveConfig",
        spec=ParamToResolveConfig,
    ) as mock_param_class:
        mock_param = mock_param_class.return_value
        mock_param.resolve.return_value = 1
        assert c.resolve_param(mock_param) == 1
        mock_param.resolve.assert_called_once_with({})


def test_resolve_param_with_other_object() -> None:
    c = AbstractConfig()
    assert c.resolve_param("value") == "value"
