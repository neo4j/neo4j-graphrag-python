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
import pytest

from neo4j_genai.exceptions import Neo4jVersionError
from neo4j_genai.retrievers.base import Retriever
from unittest.mock import MagicMock
from typing import Union, Any


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
        def search(self, *args: Any, **kwargs: Any) -> None:
            pass

    driver.execute_query.return_value = [[{"versions": db_version}], None, None]
    if expected_exception:
        with pytest.raises(expected_exception):
            MockRetriever(driver=driver)
    else:
        MockRetriever(driver=driver)
