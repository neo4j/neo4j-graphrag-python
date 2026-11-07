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
from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import neo4j
from pydantic import ValidationError

from neo4j_graphrag.decision.base import DecisionModelInterface, DecisionState
from neo4j_graphrag.exceptions import (
    DecisionModelError,
    DecisionWalkRetrievalError,
    RetrieverInitializationError,
    SearchValidationError,
)
from neo4j_graphrag.retrievers.base import Retriever
from neo4j_graphrag.types import (
    DecisionModelModel,
    DecisionWalkRetrieverModel,
    DecisionWalkSearchModel,
    Neo4jDriverModel,
    RawSearchResult,
    RetrieverResultItem,
)

logger = logging.getLogger(__name__)

DEFAULT_INSTRUCTIONS = (
    "We are walking a knowledge graph step by step to answer the question in the state. "
    "The state lists the entities we are currently standing on and the path taken so far. "
    "Choose the connection to follow next. Choose STOP only if the current entities are "
    "themselves the answer to the question."
)

DEFAULT_SEED_INSTRUCTIONS = (
    "We are about to walk a knowledge graph to answer the question in the state. "
    "Each option is a node that a text search for the question found. "
    "Choose the node the question is about: the walk to the answer starts there."
)

# The most entity names that are written into the state. A larger set ends with a count
# of the names that were left out.
MAX_NAMES_IN_STATE = 100

# Node properties that are lists longer than this, such as embeddings, are left out of
# the results.
MAX_LIST_PROPERTY_LENGTH = 32

LUCENE_SPECIAL_CHARACTERS = re.compile(r'([+\-&|!(){}\[\]^"~*?:\\/])')
LUCENE_OPERATORS = re.compile(r"\b(AND|OR|NOT)\b")

# The most words of the question that are used in the full-text search. Lucene rejects
# queries that have more than 1,024 clauses.
MAX_QUERY_WORDS = 256


def _name_expression(variable: str) -> str:
    """Cypher for a node's display name.

    The name is the first of ``$name_properties`` that the node has, or else its first
    label and its element id.
    """
    return (
        f"coalesce([v IN [p IN $name_properties | toStringOrNull({variable}[p])] "
        f"WHERE v IS NOT NULL][0], "
        f"coalesce(head(labels({variable})), 'Node') + ' ' + elementId({variable}))"
    )


SEED_QUERY = (
    "CALL db.index.fulltext.queryNodes($index_name, $query_text, {limit: $top_k}) "
    "YIELD node, score "
    f"RETURN elementId(node) AS id, {_name_expression('node')} AS name, "
    "labels(node) AS labels, score"
)

NEIGHBOURS_QUERY = (
    "MATCH (n)-[r]-(m) "
    "WHERE elementId(n) IN $node_ids AND NOT elementId(m) IN $excluded_ids "
    "RETURN elementId(n) AS from_id, type(r) AS relationship, "
    "CASE WHEN startNode(r) = n THEN 'outgoing' ELSE 'incoming' END AS direction, "
    f"elementId(m) AS to_id, {_name_expression('m')} AS to_name"
)

NODES_QUERY = (
    "MATCH (n) WHERE elementId(n) IN $node_ids "
    f"RETURN elementId(n) AS id, {_name_expression('n')} AS name, "
    "labels(n) AS labels, properties(n) AS properties"
)


def escape_lucene(text: str) -> str:
    """Make free text safe to use as a full-text (Lucene) query.

    Special characters are escaped, and ``AND``, ``OR`` and ``NOT`` are lower-cased, so
    that Lucene searches for them as words. Only the first ``MAX_QUERY_WORDS`` words
    are kept.
    """
    text = " ".join(text.split()[:MAX_QUERY_WORDS])
    text = LUCENE_OPERATORS.sub(lambda m: m.group(1).lower(), text)
    return LUCENE_SPECIAL_CHARACTERS.sub(r"\\\1", text)


def split_into_buckets(indexes: list[int], limit: int) -> list[list[int]]:
    """Split the indexes into the fewest buckets of at most ``limit`` items.

    The sizes of the buckets differ by at most one.
    """
    n_buckets = math.ceil(len(indexes) / limit)
    size, extra = divmod(len(indexes), n_buckets)
    buckets, start = [], 0
    for i in range(n_buckets):
        end = start + size + (i < extra)
        buckets.append(indexes[start:end])
        start = end
    return buckets


@dataclass
class _Option:
    """One option in a step of the walk.

    An option is either STOP, or one relationship type and direction to follow.
    """

    text: str
    relationship: Optional[str] = None
    direction: Optional[str] = None
    targets: list[str] = field(default_factory=list)
    # For each target, the node that it is reached from, the relationship type and the
    # direction.
    parents: dict[str, tuple[str, str, str]] = field(default_factory=dict)


class DecisionWalkRetriever(Retriever):
    """
    Retrieves nodes by walking the graph one decision at a time. A decision model
    chooses every step, so no LLM is used to retrieve.

    1. Start. The walk starts at ``start_node_ids`` if they are given. Otherwise, a
       search of the full-text index for the question finds up to ``seed_top_k``
       candidate nodes, and the decision model chooses the node that the question is
       about.
    2. Step. The relationships of the current nodes are grouped into options, with one
       option for each relationship type and direction. Option 0 is always STOP. The
       decision model returns a probability for every option, and the walk follows the
       chosen relationships to all of their nodes. The walk never goes back to the
       start nodes.
    3. Stop. The walk ends when the model chooses STOP, when there is nothing left to
       follow, or after ``max_steps`` steps.

    If a step has more options than the model accepts in one call, the step is decided
    in rounds, which are described in the user guide. If the chosen relationships lead
    to more than ``max_frontier_nodes`` nodes, the walk continues from the first
    ``max_frontier_nodes`` of them, in order of their names.

    The search returns the nodes that the walk ended on. Each result has the node's
    labels and properties, and the chain of relationships that led to it from the start.
    The metadata records every decision of the walk, with the probability of every
    option, the model versions that answered and the billed cost.

    Example:

    .. code-block:: python

        import neo4j
        from neo4j_graphrag.decision import OpenRouterDecisionModel
        from neo4j_graphrag.retrievers import DecisionWalkRetriever

        driver = neo4j.GraphDatabase.driver(URI, auth=AUTH)
        decision_model = OpenRouterDecisionModel("liquid/d1-ultra")

        retriever = DecisionWalkRetriever(
            driver, decision_model, fulltext_index_name="entity-names"
        )
        retriever.search(query_text="Who directed the movies Tom Hanks starred in?")

    Args:
        driver (neo4j.Driver): The Neo4j Python driver.
        decision_model (DecisionModelInterface): The decision model that chooses each step.
        fulltext_index_name (Optional[str]): The full-text index that is used to find the
            start node, if ``start_node_ids`` isn't passed to the search.
        node_name_properties (Optional[list[str]]): The properties that hold a node's
            name, in order of preference. Defaults to ``["name", "title", "id"]``. A node
            that has none of them is named by its first label and its element id.
        max_steps (int): The longest walk, in steps. Defaults to 3.
        seed_top_k (int): The number of candidate start nodes that are shown to the
            decision model. Defaults to 10.
        max_names_in_option (int): The most node names that are written into the text of
            one option. A count of the remaining names follows them. Defaults to 10.
        max_result_nodes (int): The most end nodes that are returned, in order of their
            names. Defaults to 100. The ``result_nodes_total`` key of the metadata gives
            the full count.
        max_frontier_nodes (int): The most nodes that one step moves to. The limit keeps
            the work small on graphs with nodes that have many relationships. Defaults
            to 1,000.
        instructions (Optional[str]): The instructions that the decision model gets at
            each step of the walk.
        seed_instructions (Optional[str]): The instructions that the decision model gets
            when it chooses the start node.
        result_formatter (Optional[Callable[[neo4j.Record], RetrieverResultItem]]): Custom
            function to transform a neo4j.Record to a RetrieverResultItem. The record has
            the keys ``node_id``, ``name``, ``labels``, ``properties`` and ``path``.
        neo4j_database (Optional[str]): The name of the Neo4j database. If not provided,
            this defaults to the server's default database ("neo4j" by default).

    Raises:
        RetrieverInitializationError: If validation of the input arguments fail.
    """

    def __init__(
        self,
        driver: neo4j.Driver,
        decision_model: DecisionModelInterface,
        fulltext_index_name: Optional[str] = None,
        node_name_properties: Optional[list[str]] = None,
        max_steps: int = 3,
        seed_top_k: int = 10,
        max_names_in_option: int = 10,
        max_result_nodes: int = 100,
        max_frontier_nodes: int = 1000,
        instructions: Optional[str] = None,
        seed_instructions: Optional[str] = None,
        result_formatter: Optional[
            Callable[[neo4j.Record], RetrieverResultItem]
        ] = None,
        neo4j_database: Optional[str] = None,
    ) -> None:
        try:
            validated_data = DecisionWalkRetrieverModel(
                driver_model=Neo4jDriverModel(driver=driver),
                decision_model_model=DecisionModelModel(decision_model=decision_model),
                fulltext_index_name=fulltext_index_name,
                **(
                    {"node_name_properties": node_name_properties}
                    if node_name_properties is not None
                    else {}
                ),
                max_steps=max_steps,
                seed_top_k=seed_top_k,
                max_names_in_option=max_names_in_option,
                max_result_nodes=max_result_nodes,
                max_frontier_nodes=max_frontier_nodes,
                instructions=instructions,
                seed_instructions=seed_instructions,
                result_formatter=result_formatter,
                neo4j_database=neo4j_database,
            )
        except ValidationError as e:
            raise RetrieverInitializationError(e.errors()) from e

        super().__init__(
            validated_data.driver_model.driver, validated_data.neo4j_database
        )
        self.decision_model = validated_data.decision_model_model.decision_model
        self.fulltext_index_name = validated_data.fulltext_index_name
        self.node_name_properties = validated_data.node_name_properties
        self.max_steps = validated_data.max_steps
        self.seed_top_k = validated_data.seed_top_k
        self.max_names_in_option = validated_data.max_names_in_option
        self.max_result_nodes = validated_data.max_result_nodes
        self.max_frontier_nodes = validated_data.max_frontier_nodes
        self.instructions = validated_data.instructions or DEFAULT_INSTRUCTIONS
        self.seed_instructions = (
            validated_data.seed_instructions or DEFAULT_SEED_INSTRUCTIONS
        )
        self.result_formatter = validated_data.result_formatter

    def get_search_results(
        self,
        query_text: str,
        start_node_ids: Optional[list[str]] = None,
        max_steps: Optional[int] = None,
    ) -> RawSearchResult:
        """Walk the graph from the start node(s), one decision at a time, to the nodes
        that answer the question.

        Args:
            query_text (str): The question.
            start_node_ids (Optional[list[str]]): Element ids of the nodes to start from.
                If not given, the start node is found with the full-text index.
            max_steps (Optional[int]): The longest walk for this search. It overrides the
                retriever's setting.

        Raises:
            SearchValidationError: If validation of the input arguments fail.
            DecisionWalkRetrievalError: If no start node is given and the retriever
                has no full-text index.
            DecisionModelError: If the decision model returns a choice that is not an option.

        Returns:
            RawSearchResult: One record for each node that the walk ended on. The metadata
            describes the walk.
        """
        try:
            validated_data = DecisionWalkSearchModel(
                query_text=query_text,
                start_node_ids=start_node_ids,
                max_steps=max_steps,
            )
        except ValidationError as e:
            raise SearchValidationError(e.errors()) from e
        query_text = validated_data.query_text
        steps_allowed = validated_data.max_steps or self.max_steps
        decisions: list[dict[str, Any]] = []

        if validated_data.start_node_ids is not None:
            starts = self._fetch_nodes(validated_data.start_node_ids)
        else:
            starts = self._find_start(query_text, decisions)
        names = {node["id"]: node["name"] for node in starts}
        start_ids = sorted(names, key=lambda node_id: (names[node_id], node_id))

        current, current_total = start_ids, len(start_ids)
        parents: list[dict[str, tuple[str, str, str]]] = []
        path_so_far: list[str] = []
        steps: list[str] = []
        truncated_steps: list[int] = []
        stopped_by = "no_start_node" if not start_ids else "max_steps"
        for hop in range(steps_allowed if start_ids else 0):
            options = self._build_options(current, start_ids, names)
            if len(options) == 1:
                stopped_by = "forced_stop"
                break
            state = {
                "question": query_text,
                "current_entities": self._names_in_state(current, names),
                "path_so_far": path_so_far,
                "hop": hop,
            }
            chosen = options[
                self._choose(
                    state,
                    [option.text for option in options],
                    self.instructions,
                    decisions,
                    has_stop=True,
                    context={"kind": "step", "hop": hop},
                )
            ]
            if chosen.relationship is None:
                stopped_by = "model_stop"
                break
            path_so_far = path_so_far + [self._describe_step(current, chosen, names)]
            steps.append(
                chosen.relationship + (">" if chosen.direction == "outgoing" else "<")
            )
            parents.append(chosen.parents)
            current = chosen.targets[: self.max_frontier_nodes]
            current_total = len(chosen.targets)
            if current_total > len(current):
                truncated_steps.append(hop)
                logger.warning(
                    "DecisionWalkRetriever: step %d reached %d nodes, so the walk "
                    "continues from the first %d",
                    hop,
                    current_total,
                    len(current),
                )

        records = self._build_records(current[: self.max_result_nodes], parents, names)
        costs = [
            d["usage"]["cost"]
            for d in decisions
            if d["usage"] and d["usage"]["cost"] is not None
        ]
        return RawSearchResult(
            records=records,
            metadata={
                "start_nodes": [names[node_id] for node_id in start_ids],
                "start_node_ids": start_ids,
                "path": path_so_far,
                "steps": steps,
                "stopped_by": stopped_by,
                "result_nodes_total": current_total,
                "truncated_steps": truncated_steps,
                "decisions": decisions,
                "decision_calls": len(decisions),
                "decision_input_tokens": sum(
                    (d["usage"] or {}).get("input_tokens") or 0 for d in decisions
                ),
                "decision_cost": sum(costs) if costs else None,
                "decision_models": sorted(
                    {d["model"] for d in decisions if d["model"]}
                ),
            },
        )

    def default_record_formatter(self, record: neo4j.Record) -> RetrieverResultItem:
        """Write one line for each node, with its name and labels, the path that reached
        it and its other properties."""
        properties = ", ".join(
            f"{key}: {value}"
            for key, value in sorted(record["properties"].items())
            if str(value) != record["name"]
        )
        labels = f" (:{':'.join(record['labels'])})" if record["labels"] else ""
        content = f"{record['name']}{labels} | path: {record['path']}"
        if properties:
            content += f" | properties: {properties}"
        return RetrieverResultItem(
            content=content,
            metadata={
                "node_id": record["node_id"],
                "name": record["name"],
                "labels": record["labels"],
                "path": record["path"],
            },
        )

    # ---- start --------------------------------------------------------------------------

    def _find_start(
        self, query_text: str, decisions: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Find candidate start nodes with the full-text index, and ask the decision model
        to choose one of them."""
        if self.fulltext_index_name is None:
            raise DecisionWalkRetrievalError(
                "There is no start node. Pass start_node_ids to the search, "
                "or pass a fulltext_index_name to the retriever."
            )
        if not query_text.strip():
            return []
        records, _, _ = self.driver.execute_query(
            SEED_QUERY,
            {
                "index_name": self.fulltext_index_name,
                "query_text": escape_lucene(query_text),
                "top_k": self.seed_top_k,
                "name_properties": self.node_name_properties,
            },
            database_=self.neo4j_database,
            routing_=neo4j.RoutingControl.READ,
        )
        candidates = [record.data() for record in records]
        if len(candidates) <= 1:
            return candidates
        texts = [f"start at {c['name']} ({', '.join(c['labels'])})" for c in candidates]
        chosen = self._choose(
            {"question": query_text},
            texts,
            self.seed_instructions,
            decisions,
            has_stop=False,
            context={"kind": "start", "hop": None},
        )
        return [candidates[chosen]]

    # ---- one step -----------------------------------------------------------------------

    def _build_options(
        self, current: list[str], start_ids: list[str], names: dict[str, str]
    ) -> list[_Option]:
        """Build the options for one step.

        STOP comes first, then one option for each relationship type and direction, in
        sorted order.
        """
        records, _, _ = self.driver.execute_query(
            NEIGHBOURS_QUERY,
            {
                "node_ids": current,
                "excluded_ids": start_ids,
                "name_properties": self.node_name_properties,
            },
            database_=self.neo4j_database,
            routing_=neo4j.RoutingControl.READ,
        )
        rows = [record.data() for record in records]
        for row in rows:
            names[row["to_id"]] = row["to_name"]
        # Sorted, so the same graph always gives the same options and the same parents.
        rows.sort(
            key=lambda row: (
                row["relationship"],
                row["direction"],
                names[row["from_id"]],
                row["from_id"],
            )
        )
        groups: dict[tuple[str, str], dict[str, tuple[str, str, str]]] = {}
        for row in rows:
            group = groups.setdefault((row["relationship"], row["direction"]), {})
            group.setdefault(
                row["to_id"], (row["from_id"], row["relationship"], row["direction"])
            )

        options = [_Option(text="STOP: the answer is: " + self._shown(current, names))]
        for (relationship, direction), group in sorted(groups.items()):
            targets = sorted(group, key=lambda node_id: (names[node_id], node_id))
            options.append(
                _Option(
                    text=f"follow {relationship} ({direction}) to: "
                    + self._shown(targets, names),
                    relationship=relationship,
                    direction=direction,
                    targets=targets,
                    parents=group,
                )
            )
        return options

    def _choose(
        self,
        state: DecisionState,
        texts: list[str],
        instructions: str,
        decisions: list[dict[str, Any]],
        has_stop: bool,
        context: dict[str, Any],
    ) -> int:
        """Return the index of the chosen option.

        If there are more options than the model accepts in one call, the options are
        split into buckets of even size, and the model picks a winner in each bucket. The
        winners then go to a final round. STOP, which is option 0, only takes part in the
        final round.
        """
        limit = self.decision_model.max_options
        final_only = [0] if has_stop else []
        candidates = list(range(len(final_only), len(texts)))
        while len(final_only) + len(candidates) > limit:
            # A bucket of one option goes through to the next round without a call.
            candidates = [
                self._ask(state, texts, bucket, instructions, decisions, context)
                if len(bucket) > 1
                else bucket[0]
                for bucket in split_into_buckets(candidates, limit)
            ]
        return self._ask(
            state, texts, final_only + candidates, instructions, decisions, context
        )

    def _ask(
        self,
        state: DecisionState,
        texts: list[str],
        indexes: list[int],
        instructions: str,
        decisions: list[dict[str, Any]],
        context: dict[str, Any],
    ) -> int:
        options = {str(key): texts[i] for key, i in enumerate(indexes)}
        response = self.decision_model.decide(state, options, instructions)
        if response.choice not in options:
            raise DecisionModelError(
                f"The model chose {response.choice!r}, which is not one of the options."
            )
        logger.debug(
            "DecisionWalkRetriever %s: chose %r",
            context["kind"],
            options[response.choice],
        )
        decisions.append(
            context
            | {
                "options": [texts[i] for i in indexes],
                "option_indexes": indexes,
                "choice": response.choice,
                "probabilities": response.probabilities,
                "confidence": response.confidence,
                "model": response.model,
                "usage": response.usage.model_dump() if response.usage else None,
            }
        )
        return indexes[int(response.choice)]

    # ---- result -------------------------------------------------------------------------

    def _fetch_nodes(self, node_ids: list[str]) -> list[dict[str, Any]]:
        records, _, _ = self.driver.execute_query(
            NODES_QUERY,
            {"node_ids": node_ids, "name_properties": self.node_name_properties},
            database_=self.neo4j_database,
            routing_=neo4j.RoutingControl.READ,
        )
        return [record.data() for record in records]

    def _build_records(
        self,
        node_ids: list[str],
        parents: list[dict[str, tuple[str, str, str]]],
        names: dict[str, str],
    ) -> list[neo4j.Record]:
        nodes = {node["id"]: node for node in self._fetch_nodes(node_ids)}
        return [
            neo4j.Record(
                {
                    "node_id": node_id,
                    "name": names[node_id],
                    "labels": nodes[node_id]["labels"],
                    "properties": {
                        key: value
                        for key, value in nodes[node_id]["properties"].items()
                        if not (
                            isinstance(value, list)
                            and len(value) > MAX_LIST_PROPERTY_LENGTH
                        )
                    },
                    "path": self._path_to(node_id, parents, names),
                }
            )
            for node_id in node_ids
            if node_id in nodes
        ]

    @staticmethod
    def _path_to(
        node_id: str,
        parents: list[dict[str, tuple[str, str, str]]],
        names: dict[str, str],
    ) -> str:
        """Build the chain of relationships from a start node to the given node, with
        one relationship for each step."""
        path = names[node_id]
        for step in reversed(parents):
            previous, relationship, direction = step[node_id]
            arrow = (
                f"-[{relationship}]->"
                if direction == "outgoing"
                else f"<-[{relationship}]-"
            )
            path = f"{names[previous]} {arrow} {path}"
            node_id = previous
        return path

    # ---- text ---------------------------------------------------------------------------

    def _shown(self, node_ids: list[str], names: dict[str, str]) -> str:
        text = ", ".join(names[i] for i in node_ids[: self.max_names_in_option])
        if len(node_ids) > self.max_names_in_option:
            text += f" ... (+{len(node_ids) - self.max_names_in_option} more)"
        return text

    @staticmethod
    def _names_in_state(node_ids: list[str], names: dict[str, str]) -> list[str]:
        shown = [names[i] for i in node_ids[:MAX_NAMES_IN_STATE]]
        if len(node_ids) > MAX_NAMES_IN_STATE:
            shown.append(f"... (+{len(node_ids) - MAX_NAMES_IN_STATE} more)")
        return shown

    def _describe_step(
        self, current: list[str], option: _Option, names: dict[str, str]
    ) -> str:
        arrow = "->" if option.direction == "outgoing" else "<-"
        return (
            f"{{{self._shown(current, names)}}} {option.relationship} "
            f"({option.direction} {arrow}) {{{self._shown(option.targets, names)}}}"
        )
