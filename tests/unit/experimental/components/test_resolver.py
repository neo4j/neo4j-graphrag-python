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
from unittest.mock import MagicMock, call

import neo4j
import pytest
from neo4j_graphrag.experimental.components.resolver import (
    SinglePropertyExactMatchResolver,
    SpaCySemanticMatchResolver,
)
from neo4j_graphrag.experimental.components.types import ResolutionStats


@pytest.mark.asyncio
async def test_simple_resolver(driver: MagicMock) -> None:
    driver.execute_query.side_effect = [
        ([neo4j.Record({"c": 2})], None, None),
        ([neo4j.Record({"c": 1})], None, None),
    ]
    resolver = SinglePropertyExactMatchResolver(driver=driver)
    res = await resolver.run()
    assert isinstance(res, ResolutionStats)
    assert res.number_of_nodes_to_resolve == 2
    assert res.number_of_created_nodes == 1
    assert driver.execute_query.call_count == 2
    driver.execute_query.assert_has_calls(
        [call("MATCH (entity:__Entity__)  RETURN count(entity) as c", database_=None)]
    )


@pytest.mark.asyncio
async def test_simple_resolver_custom_filter(driver: MagicMock) -> None:
    driver.execute_query.side_effect = [
        ([neo4j.Record({"c": 2})], None, None),
        ([neo4j.Record({"c": 1})], None, None),
    ]
    resolver = SinglePropertyExactMatchResolver(
        driver=driver, filter_query="WHERE not entity:Resolved"
    )
    await resolver.run()
    driver.execute_query.assert_has_calls(
        [
            call(
                "MATCH (entity:__Entity__) WHERE not entity:Resolved RETURN count(entity) as c",
                database_=None,
            )
        ]
    )
    
    
@pytest.mark.asyncio
async def test_spacy_resolver_match_on_name_property(driver: MagicMock) -> None:
    driver.execute_query.side_effect = [
        (
            [
                neo4j.Record(
                    {
                        "lab": "Person",
                        "labelCluster": [
                            {"id": 1, "name": "Alice"},
                            {"id": 2, "name": "Alice"}
                        ],
                    }
                )
            ],
            None,
            None,
        ),
        (
            [neo4j.Record({"id(node)": 1})],
            None,
            None,
        ),
    ]

    resolver = SpaCySemanticMatchResolver(driver=driver)

    res = await resolver.run()
    assert isinstance(res, ResolutionStats)
    assert res.number_of_nodes_to_resolve == 2
    assert res.number_of_created_nodes == 1

    assert driver.execute_query.call_count == 2

@pytest.mark.asyncio
async def test_spacy_resolver_no_merge(driver: MagicMock) -> None:
    driver.execute_query.side_effect = [
        (
            [
                neo4j.Record(
                    {
                        "lab": "Person",
                        "labelCluster": [
                            {"id": 1, "name": "Alice"},
                            {"id": 2, "name": "Bob"}
                        ],
                    }
                )
            ],
            None,
            None,
        ),
    ]

    resolver = SpaCySemanticMatchResolver(driver=driver)

    res = await resolver.run()
    assert res.number_of_nodes_to_resolve == 2
    assert res.number_of_created_nodes == 0

    assert driver.execute_query.call_count == 1

@pytest.mark.asyncio
async def test_spacy_resolver_match_on_multiple_text_properties(
        driver: MagicMock
) -> None:
    driver.execute_query.side_effect = [
        (
            [
                neo4j.Record(
                    {
                        "lab": "Person",
                        "labelCluster": [
                            {
                                "id": 10,
                                "name": "John Smith",
                                "ssn": "23-45-6789"
                            },
                            {
                                "id": 11,
                                "name": "Jonathan Smith",
                                "ssn": "23-45-6789"
                            }
                        ],
                    }
                )
            ],
            None,
            None,
        ),
        (
            [neo4j.Record({"id(node)": 10})],
            None,
            None,
        ),
    ]

    resolver = SpaCySemanticMatchResolver(
        driver=driver,
        resolve_properties=["name", "ssn"]
    )
    res = await resolver.run()
    assert isinstance(res, ResolutionStats)
    assert res.number_of_nodes_to_resolve == 2
    assert res.number_of_created_nodes == 1

    assert driver.execute_query.call_count == 2
