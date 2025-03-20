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
import abc
import logging
from itertools import combinations
from typing import Any, Optional, List

import neo4j
import numpy as np
from numpy.typing import NDArray
import spacy
from spacy.language import Language
from spacy.cli.download import download as spacy_download

from neo4j_graphrag.experimental.components.types import ResolutionStats
from neo4j_graphrag.experimental.pipeline import Component
from neo4j_graphrag.utils import driver_config

logger = logging.getLogger(__name__)


class EntityResolver(Component, abc.ABC):
    """Entity resolution base class

    Args:
        driver (neo4j.Driver): The Neo4j driver to connect to the database.
        filter_query (Optional[str]): Cypher query to select the entities to resolve. By default, all nodes with __Entity__ label are used
    """

    def __init__(
        self,
        driver: neo4j.Driver,
        filter_query: Optional[str] = None,
    ) -> None:
        self.driver = driver_config.override_user_agent(driver)
        self.filter_query = filter_query

    @abc.abstractmethod
    async def run(self, *args: Any, **kwargs: Any) -> ResolutionStats:
        pass


class SinglePropertyExactMatchResolver(EntityResolver):
    """Resolve entities with same label and exact same property (default is "name").

    Args:
        driver (neo4j.Driver): The Neo4j driver to connect to the database.
        filter_query (Optional[str]): To reduce the resolution scope, add a Cypher WHERE clause.
        resolve_property (str): The property that will be compared (default: "name"). If values match exactly, entities are merged.
        neo4j_database (Optional[str]): The name of the Neo4j database. If not provided, this defaults to the server's default database ("neo4j" by default) (`see reference to documentation <https://neo4j.com/docs/operations-manual/current/database-administration/#manage-databases-default>`_).

    Example:

    .. code-block:: python

        from neo4j import GraphDatabase
        from neo4j_graphrag.experimental.components.resolver import SinglePropertyExactMatchResolver

        URI = "neo4j://localhost:7687"
        AUTH = ("neo4j", "password")
        DATABASE = "neo4j"

        driver = GraphDatabase.driver(URI, auth=AUTH)
        resolver = SinglePropertyExactMatchResolver(driver=driver, neo4j_database=DATABASE)
        await resolver.run()  # no expected parameters

    """

    def __init__(
        self,
        driver: neo4j.Driver,
        filter_query: Optional[str] = None,
        resolve_property: str = "name",
        neo4j_database: Optional[str] = None,
    ) -> None:
        super().__init__(driver, filter_query)
        self.resolve_property = resolve_property
        self.neo4j_database = neo4j_database

    async def run(self) -> ResolutionStats:
        """Resolve entities based on the following rule:
        For each entity label, entities with the same 'resolve_property' value
        (exact match) are grouped into a single node:

        - Properties: the property from the first node will remain if already set, otherwise the first property in list will be written.
        - Relationships: merge relationships with same type and target node.

        See apoc.refactor.mergeNodes documentation for more details.
        """
        match_query = "MATCH (entity:__Entity__) "
        if self.filter_query:
            match_query += self.filter_query
        stat_query = f"{match_query} RETURN count(entity) as c"
        records, _, _ = self.driver.execute_query(
            stat_query,
            database_=self.neo4j_database,
        )
        number_of_nodes_to_resolve = records[0].get("c")
        if number_of_nodes_to_resolve == 0:
            return ResolutionStats(
                number_of_nodes_to_resolve=0,
            )
        merge_nodes_query = (
            f"{match_query} "
            f"WITH entity, entity.{self.resolve_property} as prop "
            # keep only entities for which the resolve_property (name) is not null
            "WITH entity, prop WHERE prop IS NOT NULL "
            # will check the property for each of the entity labels,
            # except the reserved ones __Entity__ and __KGBuilder__
            "UNWIND labels(entity) as lab  "
            "WITH lab, prop, entity WHERE NOT lab IN ['__Entity__', '__KGBuilder__'] "
            # aggregate based on property value and label
            # collect all entities with exact same property and label
            # in the 'entities' list
            "WITH prop, lab, collect(entity) AS entities "
            # merge all entities into a single node
            # * merge relationships: if the merged entities have a relationship of same
            # type to the same target node, these relationships are merged
            # otherwise relationships are just attached to the newly created node
            # * properties: if the two entities have the same property key with
            # different values, only one of them is kept in the created node
            "CALL apoc.refactor.mergeNodes(entities,{ "
            " properties:'discard', "
            " mergeRels:true "
            "}) "
            "YIELD node "
            "RETURN count(node) as c "
        )
        records, _, _ = self.driver.execute_query(
            merge_nodes_query, database_=self.neo4j_database
        )
        number_of_created_nodes = records[0].get("c")
        return ResolutionStats(
            number_of_nodes_to_resolve=number_of_nodes_to_resolve,
            number_of_created_nodes=number_of_created_nodes,
        )


class SpaCySemanticMatchResolver(EntityResolver):
    """
    Resolve entities with same label and similar set of textual properties (default is
    ["name"]) based on spaCy's static embeddings and cosine similarities.

    Args:
        driver (neo4j.Driver): The Neo4j driver to connect to the database.
        filter_query (Optional[str]): Optional Cypher WHERE clause to reduce the resolution scope.
        resolve_properties (Optional[List[str]]): The list of properties to consider for embeddings Defaults to ["name"].
        similarity_threshold (float): The similarity threshold above which nodes are merged. Defaults to 0.8.
        spacy_model (str): The name of the spaCy model to load. Defaults to "en_core_web_lg".
        neo4j_database (Optional[str]): The name of the Neo4j database. If not provided, this defaults to the server's default database ("neo4j" by default) (`see reference to documentation <https://neo4j.com/docs/operations-manual/current/database-administration/#manage-databases-default>`_).

    Example:

    .. code-block:: python

        from neo4j import GraphDatabase
        from neo4j_graphrag.experimental.components.resolver import SinglePropertyExactMatchResolver

        URI = "neo4j://localhost:7687"
        AUTH = ("neo4j", "password")
        DATABASE = "neo4j"

        driver = GraphDatabase.driver(URI, auth=AUTH)
        resolver = SinglePropertyExactMatchResolver(driver=driver, neo4j_database=DATABASE)
        await resolver.run()  # no expected parameters

    """

    def __init__(
        self,
        driver: neo4j.Driver,
        filter_query: Optional[str] = None,
        resolve_properties: Optional[List[str]] = None,
        similarity_threshold: float = 0.8,
        spacy_model: str = "en_core_web_lg",
        neo4j_database: Optional[str] = None,
    ) -> None:
        super().__init__(driver, filter_query)
        self.resolve_properties = resolve_properties or ["name"]
        self.similarity_threshold = similarity_threshold
        self.neo4j_database = neo4j_database
        self.nlp = self._load_or_download_spacy_model(spacy_model)

    async def run(self) -> ResolutionStats:
        """Resolve entities based on the following rules:
         For each entity label, entities with similar 'resolve_properties'
        (cosine similarity on embedding vectors) are merged into a single node.

        See apoc.refactor.mergeNodes documentation for more details.
        """
        match_query = "MATCH (entity:__Entity__)"
        if self.filter_query:
            match_query += f" {self.filter_query}"

        # generate a dynamic map of requested properties, e.g. "name: entity.name, description: entity.description, ..."
        props_map_list = [f"{prop}: entity.{prop}" for prop in self.resolve_properties]
        props_map = ", ".join(props_map_list)

        # Cypher query:
        # matches extracted entities
        # filters entities if filter_query is provided
        # unwinds labels to skip reserved ones
        # collects all properties needed for embeddings
        query = f"""
            {match_query}
            UNWIND labels(entity) AS lab
            WITH lab, entity
            WHERE NOT lab IN ['__Entity__', '__KGBuilder__']
            WITH lab, collect({{ id: id(entity), {props_map} }}) AS labelCluster
            RETURN lab, labelCluster
        """

        records, _, _ = self.driver.execute_query(query, database_=self.neo4j_database)

        total_entities_embedded = 0
        total_merged_nodes = 0

        # for each row, 'lab' is the label, 'labelCluster' is a list of dicts (id + textual properties)
        for row in records:
            entities = row["labelCluster"]

            # build node embeddings
            node_embeddings = {}
            for ent in entities:
                # concatenate all textual properties (if non-null) into a single string
                texts = [
                    str(ent[p]) for p in self.resolve_properties if p in ent and ent[p]
                ]
                combined_text = " ".join(texts).strip()
                if combined_text:
                    node_embeddings[ent["id"]] = self.nlp(combined_text).vector
            total_entities_embedded += len(node_embeddings)

            # identify pairs to merge
            pairs_to_merge = []
            for (id1, emb1), (id2, emb2) in combinations(node_embeddings.items(), 2):
                sim = self._cosine_similarity(
                    np.asarray(emb1, dtype=np.float64),
                    np.asarray(emb2, dtype=np.float64),
                )
                if sim >= self.similarity_threshold:
                    pairs_to_merge.append({id1, id2})

            # consolidate overlapping sets of node IDs
            resolved_sets = self._consolidate_sets(pairs_to_merge)

            # perform merges in the db using APOC
            merged_count = 0
            for node_id_set in resolved_sets:
                if len(node_id_set) > 1:
                    merge_query = (
                        "MATCH (n) WHERE id(n) IN $ids "
                        "WITH collect(n) AS nodes "
                        "CALL apoc.refactor.mergeNodes(nodes, {properties: 'discard', mergeRels: true}) "
                        "YIELD node RETURN id(node)"
                    )
                    result, _, _ = self.driver.execute_query(
                        merge_query,
                        {"ids": list(node_id_set)},
                        database_=self.neo4j_database,
                    )
                    merged_count += len(result)

            total_merged_nodes += merged_count

        return ResolutionStats(
            number_of_nodes_to_resolve=total_entities_embedded,
            number_of_created_nodes=total_merged_nodes,
        )

    @staticmethod
    def _consolidate_sets(pairs: List[set[str]]) -> List[set[str]]:
        """Consolidate overlapping sets of node pairs into unique sets."""
        consolidated: List[set[str]] = []
        for pair in pairs:
            merged = False
            for cons in consolidated:
                # if there is any intersection, unify them
                if pair & cons:
                    cons.update(pair)
                    merged = True
                    break
            if not merged:
                consolidated.append(set(pair))
        return consolidated

    @staticmethod
    def _cosine_similarity(
        vec1: NDArray[np.float64], vec2: NDArray[np.float64]
    ) -> float:
        """Calculate cosine similarity between two embedding vectors."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if not norm1 or not norm2:
            return 0.0
        return float(dot_product / (norm1 * norm2))

    @staticmethod
    def _load_or_download_spacy_model(model_name: str) -> Language:
        """
        Attempt to load the specified spaCy model by name.
        If not installed, automatically download and then load it.
        """
        try:
            return spacy.load(model_name)
        except OSError as e:
            # handling cases where the spaCy model is not yet downloaded:
            if "doesn't seem to be a Python package or a valid path" in str(e):
                logger.info(f"Model '{model_name}' not found. Downloading...")
                spacy_download(model_name)
                return spacy.load(model_name)
            else:
                raise e
