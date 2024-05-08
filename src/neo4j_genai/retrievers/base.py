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
from abc import ABC, abstractmethod
from typing import Any

import neo4j


class Retriever(ABC):
    """
    Abstract class for Neo4j retrievers
    """

    def __init__(self, driver: neo4j.Driver):
        self.driver = driver
        self._verify_version()

    def _verify_version(self) -> None:
        """
        Check if the connected Neo4j database version supports vector indexing.

        Queries the Neo4j database to retrieve its version and compares it
        against a target version (5.18.1) that is known to support vector
        indexing. Raises a ValueError if the connected Neo4j version is
        not supported.
        """
        records, _, _ = self.driver.execute_query("CALL dbms.components()")
        version = records[0]["versions"][0]

        if "aura" in version:
            version_tuple = (
                *tuple(map(int, version.split("-")[0].split("."))),
                0,
            )
            target_version = (5, 18, 0)
        else:
            version_tuple = tuple(map(int, version.split(".")))
            target_version = (5, 18, 1)

        if version_tuple < target_version:
            raise ValueError(
                "This package only supports Neo4j version 5.18.1 or greater"
            )

    @abstractmethod
    def search(self, *args, **kwargs) -> Any:
        pass
