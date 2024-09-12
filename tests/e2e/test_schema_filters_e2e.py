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
from neo4j import Driver
from neo4j_graphrag.schema import (
    BASE_ENTITY_LABEL,
    EXCLUDED_LABELS,
    EXCLUDED_RELS,
    NODE_PROPERTIES_QUERY,
    REL_PROPERTIES_QUERY,
    REL_QUERY,
    query_database,
)


@pytest.mark.usefixtures("setup_neo4j_for_schema_query_with_excluded_labels")
def test_filtering_labels_node_properties(driver: Driver) -> None:
    node_properties = [
        data["output"]
        for data in query_database(
            driver,
            NODE_PROPERTIES_QUERY,
            params={"EXCLUDED_LABELS": EXCLUDED_LABELS},
        )
    ]

    assert node_properties == []


@pytest.mark.usefixtures("setup_neo4j_for_schema_query_with_excluded_labels")
def test_filtering_labels_relationship_properties(driver: Driver) -> None:
    relationship_properties = [
        data["output"]
        for data in query_database(
            driver, REL_PROPERTIES_QUERY, params={"EXCLUDED_LABELS": EXCLUDED_RELS}
        )
    ]

    assert relationship_properties == []


@pytest.mark.usefixtures("setup_neo4j_for_schema_query_with_excluded_labels")
def test_filtering_labels_relationships(driver: Driver) -> None:
    relationships = [
        data["output"]
        for data in query_database(
            driver,
            REL_QUERY,
            params={"EXCLUDED_LABELS": EXCLUDED_LABELS + [BASE_ENTITY_LABEL]},
        )
    ]

    assert relationships == []
