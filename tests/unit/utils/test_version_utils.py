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
from unittest.mock import MagicMock

import pytest
from neo4j_graphrag.utils.version_utils import (
    get_version,
    has_vector_index_support,
    has_metadata_filtering_support,
    is_version_5_23_or_above,
)


@pytest.mark.parametrize(
    "db_version,expected_version",
    [
        (["5.18-aura"], ((5, 18, 0), True, True)),
        (["5.3-aura"], ((5, 3, 0), True, True)),
        (["5.19.0"], ((5, 19, 0), False, True)),
        (["4.3.5"], ((4, 3, 5), False, True)),
        (["5.23.0-6698"], ((5, 23, 0), False, True)),
        (["2025.01.0"], ((2025, 1, 0), False, True)),
        (["2025.01-aura"], ((2025, 1, 0), True, True)),
    ],
)
def test_get_version(
    driver: MagicMock,
    db_version: list[str],
    expected_version: tuple[tuple[int, ...], bool, bool],
) -> None:
    """
    Verifies that the get_version function correctly parses the database
    version and identifies whether the database is hosted on the Aura platform.
    """
    driver.execute_query.return_value = [
        [{"versions": db_version, "edition": "enterprise"}],
        None,
        None,
    ]
    assert get_version(driver) == expected_version, f"Failed test case: {db_version}"


@pytest.mark.parametrize(
    "version_tuple,expected_result",
    [
        ((5, 22, 0), False),
        ((5, 23, 0), True),
        ((2025, 1, 0), True),
    ],
)
def test_is_version_5_23_or_above(
    version_tuple: tuple[int, ...], expected_result: bool
) -> None:
    """
    Ensures that the is_version_5_23_or_above function accurately determines if
    a given version is 5.23 or higher.
    """
    assert (
        is_version_5_23_or_above(version_tuple) == expected_result
    ), f"Failed test case: {version_tuple}"


@pytest.mark.parametrize(
    "version_tuple,expected_result",
    [
        ((5, 10, 0), False),
        ((5, 11, 0), True),
        ((2025, 1, 0), True),
    ],
)
def test_has_vector_index_support(
    version_tuple: tuple[int, ...], expected_result: bool
) -> None:
    """
    Tests the has_vector_index_support function to confirm it correctly
    identifies if the given version and platform support vector indexing.
    """
    assert (
        has_vector_index_support(version_tuple) == expected_result
    ), f"Failed test case: {version_tuple}"


@pytest.mark.parametrize(
    "version_tuple,is_aura,expected_result",
    [
        ((5, 18, 0), True, True),
        ((5, 18, 0), False, False),
        ((5, 18, 1), True, True),
        ((5, 18, 1), False, True),
        ((2025, 1, 0), True, True),
        ((2025, 1, 0), False, True),
    ],
)
def test_has_metadata_filtering_support(
    version_tuple: tuple[int, ...], is_aura: bool, expected_result: bool
) -> None:
    """
    Tests the has_metadata_filtering_support function to confirm it correctly
    identifies if the given version and platform support vector index metadata filtering.
    """
    assert (
        has_metadata_filtering_support(version_tuple, is_aura) == expected_result
    ), f"Failed test case: {version_tuple}, is_aura: {is_aura}"
