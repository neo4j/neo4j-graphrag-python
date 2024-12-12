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
from unittest.mock import patch

import pytest
from neo4j_graphrag.experimental.pipeline.config.param_resolver import (
    ParamFromEnvConfig,
    ParamFromKeyConfig,
)


@patch.dict(os.environ, {"MY_KEY": "my_value"}, clear=True)
def test_env_param_config_happy_path() -> None:
    resolver = ParamFromEnvConfig(var_="MY_KEY")
    assert resolver.resolve({}) == "my_value"


@patch.dict(os.environ, {}, clear=True)
def test_env_param_config_missing_env_var() -> None:
    resolver = ParamFromEnvConfig(var_="MY_KEY")
    assert resolver.resolve({}) is None


def test_config_key_param_simple_key() -> None:
    resolver = ParamFromKeyConfig(key_="my_key")
    assert resolver.resolve({"my_key": "my_value"}) == "my_value"


def test_config_key_param_missing_key() -> None:
    resolver = ParamFromKeyConfig(key_="my_key")
    with pytest.raises(KeyError):
        resolver.resolve({})


def test_config_complex_key_param() -> None:
    resolver = ParamFromKeyConfig(key_="my_key.my_sub_key")
    assert resolver.resolve({"my_key": {"my_sub_key": "value"}}) == "value"


def test_config_complex_key_param_missing_subkey() -> None:
    resolver = ParamFromKeyConfig(key_="my_key.my_sub_key")
    with pytest.raises(KeyError):
        resolver.resolve({"my_key": {}})
