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
"""Runs the Cypher of the DecisionWalkRetriever on a real Neo4j, with a scripted decision
model. The fixture creates a small graph, then removes the nodes that it created and
its full-text index."""

from typing import Any, Generator

import neo4j
import pytest

from neo4j_graphrag.decision import (
    DecisionModelInterface,
    DecisionResponse,
    DecisionState,
)
from neo4j_graphrag.retrievers import DecisionWalkRetriever

LABEL = "DecisionWalkE2E"
INDEX = "decision_walk_e2e_names"


class ScriptedDecisionModel(DecisionModelInterface):
    """Chooses the option whose text starts with the next scripted prefix."""

    def __init__(self, script: list[str]):
        super().__init__(model_name="scripted")
        self.script = list(script)
        self.calls: list[dict[str, str]] = []

    def decide(
        self, state: DecisionState, options: dict[str, str], instructions: str
    ) -> DecisionResponse:
        self.calls.append(options)
        prefix = self.script.pop(0)
        choice = next(k for k, text in options.items() if text.startswith(prefix))
        return DecisionResponse(
            choice=choice, probabilities={k: float(k == choice) for k in options}
        )

    async def adecide(
        self, state: DecisionState, options: dict[str, str], instructions: str
    ) -> DecisionResponse:
        return self.decide(state, options, instructions)


@pytest.fixture(scope="module")
def walk_graph(driver: neo4j.Driver) -> Generator[dict[str, str], Any, Any]:
    """Two films by one director, which share a genre and are linked to each other. The
    director has a relationship to themself, and the genre tags a node that has no label
    and no name property."""
    records, _, _ = driver.execute_query(
        f"""
        CREATE (alpha:{LABEL} {{title: 'Alpha'}}),
               (beta:{LABEL} {{title: 'Beta'}}),
               (pat:{LABEL} {{name: 'Pat Doe'}}),
               (noir:{LABEL} {{name: 'Noir', embedding: range(1, 64)}}),
               (untyped {{code: 'x'}}),
               (alpha)-[:DIRECTED_BY]->(pat), (beta)-[:DIRECTED_BY]->(pat),
               (alpha)-[:HAS_GENRE]->(noir), (beta)-[:HAS_GENRE]->(noir),
               (alpha)-[:SEQUEL_OF]->(beta), (pat)-[:KNOWS]->(pat),
               (noir)-[:TAGGED]->(untyped)
        RETURN elementId(alpha) AS alpha, elementId(beta) AS beta, elementId(pat) AS pat,
               elementId(noir) AS noir, elementId(untyped) AS untyped
        """
    )
    ids = dict(records[0])
    driver.execute_query(
        f"CREATE FULLTEXT INDEX {INDEX} IF NOT EXISTS "
        f"FOR (n:{LABEL}) ON EACH [n.name, n.title]"
    )
    driver.execute_query("CALL db.awaitIndexes(300)")
    yield ids
    driver.execute_query(
        "MATCH (n) WHERE elementId(n) IN $ids DETACH DELETE n",
        ids=list(ids.values()),
    )
    driver.execute_query(f"DROP INDEX {INDEX} IF EXISTS")


def test_decision_walk_e2e_walk(
    driver: neo4j.Driver, walk_graph: dict[str, str]
) -> None:
    model = ScriptedDecisionModel(
        ["follow DIRECTED_BY", "follow HAS_GENRE", "follow TAGGED", "STOP"]
    )
    retriever = DecisionWalkRetriever(driver, model, max_steps=4)
    result = retriever.search(
        query_text="What tags the genre of Pat Doe's films?",
        start_node_ids=[walk_graph["pat"]],
    )

    # Step 0. The self-loop leads back to the start node, so it isn't offered.
    assert model.calls[0] == {
        "0": "STOP: the answer is: Pat Doe",
        "1": "follow DIRECTED_BY (incoming) to: Alpha, Beta",
    }
    # Step 1. The relationship between the two current films is offered in both
    # directions, and Noir is offered once, although both films lead to it.
    assert model.calls[1] == {
        "0": "STOP: the answer is: Alpha, Beta",
        "1": "follow HAS_GENRE (outgoing) to: Noir",
        "2": "follow SEQUEL_OF (incoming) to: Alpha",
        "3": "follow SEQUEL_OF (outgoing) to: Beta",
    }
    untyped_name = f"Node {walk_graph['untyped']}"
    assert model.calls[2]["2"] == f"follow TAGGED (outgoing) to: {untyped_name}"

    metadata = result.metadata or {}
    assert metadata["stopped_by"] == "model_stop"
    assert metadata["steps"] == ["DIRECTED_BY<", "HAS_GENRE>", "TAGGED>"]
    # The untyped node has no labels, and its only property is not a name.
    assert [item.content for item in result.items] == [
        f"{untyped_name} | path: Pat Doe <-[DIRECTED_BY]- Alpha -[HAS_GENRE]-> Noir "
        f"-[TAGGED]-> {untyped_name} | properties: code: x"
    ]


def test_decision_walk_e2e_drops_long_list_properties(
    driver: neo4j.Driver, walk_graph: dict[str, str]
) -> None:
    retriever = DecisionWalkRetriever(driver, ScriptedDecisionModel(["STOP"]))
    result = retriever.search(query_text="q", start_node_ids=[walk_graph["noir"]])
    assert [item.content for item in result.items] == [
        "Noir (:DecisionWalkE2E) | path: Noir"
    ]


def test_decision_walk_e2e_fulltext_start(
    driver: neo4j.Driver, walk_graph: dict[str, str]
) -> None:
    model = ScriptedDecisionModel(["start at Beta", "follow DIRECTED_BY", "STOP"])
    retriever = DecisionWalkRetriever(driver, model, fulltext_index_name=INDEX)
    result = retriever.search(query_text="Who directed [Beta] AND [Alpha]?")

    assert set(model.calls[0].values()) == {
        "start at Alpha (DecisionWalkE2E)",
        "start at Beta (DecisionWalkE2E)",
    }
    assert (result.metadata or {})["start_node_ids"] == [walk_graph["beta"]]
    assert [item.content for item in result.items] == [
        "Pat Doe (:DecisionWalkE2E) | path: Beta -[DIRECTED_BY]-> Pat Doe"
    ]
