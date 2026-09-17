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
"""Tests for public exports of neo4j_graphrag.llm."""

import neo4j_graphrag.llm as llm_module


def test_base_anthropic_llm_is_exported() -> None:
    from neo4j_graphrag.llm import BaseAnthropicLLM

    assert BaseAnthropicLLM is not None
    assert "BaseAnthropicLLM" in llm_module.__all__


def test_base_openai_llm_is_exported() -> None:
    from neo4j_graphrag.llm import BaseOpenAILLM

    assert BaseOpenAILLM is not None
    assert "BaseOpenAILLM" in llm_module.__all__


def test_split_http_client_kwargs_is_exported() -> None:
    from neo4j_graphrag.llm import split_http_client_kwargs

    assert callable(split_http_client_kwargs)
    assert "split_http_client_kwargs" in llm_module.__all__
