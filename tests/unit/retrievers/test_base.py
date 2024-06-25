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
import inspect

import pytest

from typing import Union, Any
from unittest.mock import MagicMock, patch

from neo4j_genai.exceptions import Neo4jVersionError
from neo4j_genai.retrievers.base import Retriever
from neo4j_genai.types import RawSearchResult, RetrieverResult


@pytest.mark.parametrize(
    "db_version,expected_exception",
    [
        (["5.18-aura"], None),
        (["5.3-aura"], Neo4jVersionError),
        (["5.19.0"], None),
        (["4.3.5"], Neo4jVersionError),
    ],
)
def test_retriever_version_support(
    driver: MagicMock,
    db_version: list[str],
    expected_exception: Union[type[ValueError], None],
) -> None:
    class MockRetriever(Retriever):
        def _get_search_results(self, *args: Any, **kwargs: Any) -> RawSearchResult:
            return RawSearchResult(records=[])

    driver.execute_query.return_value = [[{"versions": db_version}], None, None]
    if expected_exception:
        with pytest.raises(expected_exception):
            MockRetriever(driver=driver)
    else:
        MockRetriever(driver=driver)


@patch("neo4j_genai.retrievers.base.Retriever._verify_version")
def test_retriever_search_docstring_copied(
    _verify_version_mock: MagicMock,
    driver: MagicMock,
) -> None:
    class MockRetriever(Retriever):
        def _get_search_results(self, query: str, top_k: int = 10) -> RawSearchResult:
            """My fabulous docstring"""
            return RawSearchResult(records=[])

    retriever = MockRetriever(driver=driver)
    assert retriever.search.__doc__ == "My fabulous docstring"
    signature = inspect.signature(retriever.search)
    assert "query" in signature.parameters
    query_param = signature.parameters["query"]
    assert query_param.default == query_param.empty
    assert query_param.annotation == str
    assert "top_k" in signature.parameters
    top_k_param = signature.parameters["top_k"]
    assert top_k_param.default == 10
    assert top_k_param.annotation == int


@patch("neo4j_genai.retrievers.base.Retriever._verify_version")
def test_retriever_search_docstring_unchanged(
    _verify_version_mock: MagicMock,
    driver: MagicMock,
) -> None:
    class MockRetrieverForNoise(Retriever):
        def _get_search_results(self, query: str, top_k: int = 10) -> RawSearchResult:
            """My fabulous docstring"""
            return RawSearchResult(records=[])

    class MockRetriever(Retriever):
        def _get_search_results(self, *args: Any, **kwargs: Any) -> RawSearchResult:
            return RawSearchResult(records=[])

        def search(self, query: str, top_k: int = 10) -> RetrieverResult:
            """My fabulous docstring that I do not want to be updated"""
            return RetrieverResult(items=[])

    assert MockRetrieverForNoise.search is not MockRetriever.search

    retriever = MockRetriever(driver=driver)
    assert (
        retriever.search.__doc__
        == "My fabulous docstring that I do not want to be updated"
    )
