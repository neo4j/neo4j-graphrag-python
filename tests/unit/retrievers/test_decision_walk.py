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
from typing import Any, Callable, Optional
from unittest.mock import MagicMock, patch

import neo4j
import pytest

from neo4j_graphrag.decision import (
    DecisionModelInterface,
    DecisionResponse,
    DecisionState,
    DecisionUsage,
)
from neo4j_graphrag.exceptions import (
    DecisionModelError,
    DecisionWalkRetrievalError,
    RetrieverInitializationError,
    SearchValidationError,
)
from neo4j_graphrag.retrievers import DecisionWalkRetriever
from neo4j_graphrag.retrievers.decision_walk import (
    NEIGHBOURS_QUERY,
    NODES_QUERY,
    SEED_QUERY,
    escape_lucene,
    split_into_buckets,
)
from neo4j_graphrag.types import RetrieverResultItem

# A small film graph. Each element id maps to a name, labels and other properties.
NODES: dict[str, dict[str, Any]] = {
    "inception": {"name": "Inception", "labels": ["Movie"], "released": 2010},
    "interstellar": {"name": "Interstellar", "labels": ["Movie"]},
    "titanic": {"name": "Titanic", "labels": ["Movie"]},
    "nolan": {"name": "Christopher Nolan", "labels": ["Person"]},
    "cameron": {"name": "James Cameron", "labels": ["Person"]},
    "dicaprio": {"name": "Leonardo DiCaprio", "labels": ["Person"]},
    "scifi": {"title": "Sci-Fi", "labels": ["Genre"], "embedding": [0.1] * 64},
    "lonely": {"labels": ["Thing"]},
}
EDGES = [
    ("inception", "directed_by", "nolan"),
    ("interstellar", "directed_by", "nolan"),
    ("titanic", "directed_by", "cameron"),
    ("inception", "starred_actors", "dicaprio"),
    ("titanic", "starred_actors", "dicaprio"),
    ("inception", "has_genre", "scifi"),
    ("interstellar", "has_genre", "scifi"),
]


def node_name(node_id: str, name_properties: list[str]) -> str:
    node = NODES[node_id]
    for p in name_properties:
        if p in node:
            return str(node[p])
    return f"{node['labels'][0]} {node_id}"


class FakeGraph:
    """Answers the retriever's three queries from NODES and EDGES."""

    def __init__(self, seed_hits: Optional[list[str]] = None):
        self.seed_hits = seed_hits or []
        self.queries: list[tuple[str, dict[str, Any]]] = []

    def execute_query(
        self, query: str, parameters: dict[str, Any], **kwargs: Any
    ) -> tuple[list[neo4j.Record], None, None]:
        self.queries.append((query, parameters))
        names = parameters["name_properties"]
        if query == SEED_QUERY:
            rows = [
                {
                    "id": i,
                    "name": node_name(i, names),
                    "labels": NODES[i]["labels"],
                    "score": 1.0,
                }
                for i in self.seed_hits[: parameters["top_k"]]
            ]
        elif query == NEIGHBOURS_QUERY:
            rows = []
            for a, rel, b in EDGES:
                for here, there, direction in ((a, b, "outgoing"), (b, a, "incoming")):
                    if (
                        here in parameters["node_ids"]
                        and there not in parameters["excluded_ids"]
                    ):
                        rows.append(
                            {
                                "from_id": here,
                                "relationship": rel,
                                "direction": direction,
                                "to_id": there,
                                "to_name": node_name(there, names),
                            }
                        )
        elif query == NODES_QUERY:
            rows = [
                {
                    "id": i,
                    "name": node_name(i, names),
                    "labels": NODES[i]["labels"],
                    "properties": {k: v for k, v in NODES[i].items() if k != "labels"},
                }
                for i in parameters["node_ids"]
                if i in NODES
            ]
        else:
            raise AssertionError(f"unexpected query: {query}")
        return [neo4j.Record(row) for row in rows], None, None


class ScriptedDecisionModel(DecisionModelInterface):
    """Chooses the option whose text starts with the next scripted prefix."""

    def __init__(self, script: list[str], max_options: int = 255):
        super().__init__(model_name="scripted", max_options=max_options)
        self.script = list(script)
        self.calls: list[tuple[DecisionState, dict[str, str], str]] = []

    def decide(
        self, state: DecisionState, options: dict[str, str], instructions: str
    ) -> DecisionResponse:
        self.calls.append((state, options, instructions))
        prefix = self.script.pop(0)
        choice = next(k for k, text in options.items() if text.startswith(prefix))
        return DecisionResponse(
            choice=choice,
            probabilities={k: float(k == choice) for k in options},
            confidence=0.9,
            model="scripted-v1",
            usage=DecisionUsage(input_tokens=10, cost=0.001),
        )

    async def adecide(
        self, state: DecisionState, options: dict[str, str], instructions: str
    ) -> DecisionResponse:
        return self.decide(state, options, instructions)


@pytest.fixture
def make_retriever(
    driver: MagicMock,
) -> Callable[..., tuple[DecisionWalkRetriever, FakeGraph]]:
    def make(
        script: list[str],
        seed_hits: Optional[list[str]] = None,
        max_options: int = 255,
        **kwargs: Any,
    ) -> tuple[DecisionWalkRetriever, FakeGraph]:
        graph = FakeGraph(seed_hits)
        driver.execute_query.side_effect = graph.execute_query
        with patch("neo4j_graphrag.retrievers.base.get_version") as get_version:
            get_version.return_value = ((5, 23, 0), False, False)
            retriever = DecisionWalkRetriever(
                driver, ScriptedDecisionModel(script, max_options), **kwargs
            )
        return retriever, graph

    return make


def calls_of(retriever: DecisionWalkRetriever) -> list[Any]:
    model = retriever.decision_model
    assert isinstance(model, ScriptedDecisionModel)
    return model.calls


# ---- initialization ---------------------------------------------------------------


@patch("neo4j_graphrag.retrievers.base.get_version")
def test_decision_walk_initialization(
    mock_get_version: MagicMock, driver: MagicMock
) -> None:
    mock_get_version.return_value = ((5, 23, 0), False, False)
    retriever = DecisionWalkRetriever(driver, ScriptedDecisionModel([]))
    mock_get_version.assert_called_once()
    assert retriever.node_name_properties == ["name", "title", "id"]
    assert retriever.max_steps == 3
    assert "Choose STOP" in retriever.instructions


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"decision_model": object()}, "'decide' callable method"),
        ({"decision_model": MagicMock(max_options=None)}, "integer 'max_options'"),
        ({"decision_model": MagicMock(max_options=1)}, "of at least 2"),
        ({"max_frontier_nodes": 0}, "max_frontier_nodes"),
        ({"node_name_properties": []}, "node_name_properties cannot be an empty list"),
        ({"max_steps": 0}, "max_steps"),
        ({"seed_top_k": -1}, "seed_top_k"),
    ],
)
def test_decision_walk_invalid_initialization(
    driver: MagicMock, kwargs: dict[str, Any], message: str
) -> None:
    kwargs.setdefault("decision_model", ScriptedDecisionModel([]))
    with pytest.raises(RetrieverInitializationError, match=message):
        DecisionWalkRetriever(driver, **kwargs)


def test_decision_walk_invalid_search(make_retriever: Callable[..., Any]) -> None:
    retriever, _ = make_retriever([])
    with pytest.raises(SearchValidationError, match="query_text"):
        retriever.search(query_text=42)
    with pytest.raises(SearchValidationError, match="max_steps"):
        retriever.search(query_text="q", start_node_ids=["inception"], max_steps=0)
    with pytest.raises(SearchValidationError, match="start_node_ids"):
        retriever.search(query_text="q", start_node_ids=[])


# ---- walking ----------------------------------------------------------------------


def test_decision_walk_one_step(make_retriever: Callable[..., Any]) -> None:
    retriever, _ = make_retriever(["follow directed_by", "STOP"])
    result = retriever.search(
        query_text="Who directed Inception?", start_node_ids=["inception"]
    )

    assert [item.content for item in result.items] == [
        "Christopher Nolan (:Person) | path: Inception -[directed_by]-> Christopher Nolan"
    ]
    assert result.items[0].metadata == {
        "node_id": "nolan",
        "name": "Christopher Nolan",
        "labels": ["Person"],
        "path": "Inception -[directed_by]-> Christopher Nolan",
    }
    metadata = result.metadata or {}
    assert metadata["start_nodes"] == ["Inception"]
    assert metadata["steps"] == ["directed_by>"]
    assert metadata["path"] == [
        "{Inception} directed_by (outgoing ->) {Christopher Nolan}"
    ]
    assert metadata["stopped_by"] == "model_stop"
    assert metadata["decision_calls"] == 2
    assert metadata["decision_input_tokens"] == 20
    assert metadata["decision_cost"] == pytest.approx(0.002)
    assert metadata["decision_models"] == ["scripted-v1"]
    assert metadata["__retriever"] == "DecisionWalkRetriever"


def test_decision_walk_options_and_state(make_retriever: Callable[..., Any]) -> None:
    retriever, _ = make_retriever(["follow directed_by", "STOP"])
    retriever.search(query_text="Who directed Inception?", start_node_ids=["inception"])

    (state, options, instructions), (state2, options2, _) = calls_of(retriever)
    assert state == {
        "question": "Who directed Inception?",
        "current_entities": ["Inception"],
        "path_so_far": [],
        "hop": 0,
    }
    assert options == {
        "0": "STOP: the answer is: Inception",
        "1": "follow directed_by (outgoing) to: Christopher Nolan",
        "2": "follow has_genre (outgoing) to: Sci-Fi",
        "3": "follow starred_actors (outgoing) to: Leonardo DiCaprio",
    }
    assert instructions == retriever.instructions
    # The start node is never offered again, so Nolan only leads on to Interstellar.
    assert state2["hop"] == 1
    assert state2["path_so_far"] == [
        "{Inception} directed_by (outgoing ->) {Christopher Nolan}"
    ]
    assert options2 == {
        "0": "STOP: the answer is: Christopher Nolan",
        "1": "follow directed_by (incoming) to: Interstellar",
    }


def test_decision_walk_two_steps_records_the_chain(
    make_retriever: Callable[..., Any],
) -> None:
    retriever, _ = make_retriever(
        ["follow starred_actors", "follow starred_actors (incoming)", "STOP"]
    )
    result = retriever.search(
        query_text="Which films share an actor with Inception?",
        start_node_ids=["inception"],
    )
    assert [item.content for item in result.items] == [
        "Titanic (:Movie) | path: Inception -[starred_actors]-> Leonardo DiCaprio "
        "<-[starred_actors]- Titanic"
    ]
    assert (result.metadata or {})["steps"] == ["starred_actors>", "starred_actors<"]


def test_decision_walk_stops_at_max_steps(make_retriever: Callable[..., Any]) -> None:
    retriever, _ = make_retriever(["follow has_genre"], max_steps=1)
    result = retriever.search(query_text="q", start_node_ids=["inception"])
    metadata = result.metadata or {}
    assert metadata["stopped_by"] == "max_steps"
    assert metadata["decision_calls"] == 1
    # The embedding is left out of the properties, and the name property isn't repeated.
    assert [item.content for item in result.items] == [
        "Sci-Fi (:Genre) | path: Inception -[has_genre]-> Sci-Fi"
    ]


def test_decision_walk_max_steps_per_search(make_retriever: Callable[..., Any]) -> None:
    retriever, _ = make_retriever(["follow has_genre"])
    result = retriever.search(query_text="q", start_node_ids=["inception"], max_steps=1)
    assert (result.metadata or {})["stopped_by"] == "max_steps"


def test_decision_walk_forced_stop(make_retriever: Callable[..., Any]) -> None:
    retriever, _ = make_retriever([])
    result = retriever.search(query_text="q", start_node_ids=["lonely"])
    metadata = result.metadata or {}
    assert metadata["stopped_by"] == "forced_stop"
    assert metadata["decision_calls"] == 0
    assert metadata["decision_cost"] is None
    assert [item.content for item in result.items] == [
        "Thing lonely (:Thing) | path: Thing lonely"
    ]


def test_decision_walk_stop_at_start_returns_properties(
    make_retriever: Callable[..., Any],
) -> None:
    retriever, _ = make_retriever(["STOP"])
    result = retriever.search(
        query_text="When was Inception released?", start_node_ids=["inception"]
    )
    assert [item.content for item in result.items] == [
        "Inception (:Movie) | path: Inception | properties: released: 2010"
    ]


def test_decision_walk_unknown_start_node(make_retriever: Callable[..., Any]) -> None:
    retriever, _ = make_retriever([])
    result = retriever.search(query_text="q", start_node_ids=["missing"])
    assert result.items == []
    assert (result.metadata or {})["stopped_by"] == "no_start_node"


def test_decision_walk_limits_frontier(make_retriever: Callable[..., Any]) -> None:
    retriever, _ = make_retriever(
        ["follow directed_by", "follow has_genre", "STOP"], max_frontier_nodes=1
    )
    result = retriever.search(query_text="q", start_node_ids=["nolan"])
    metadata = result.metadata or {}
    # The walk only continues from Inception, the first node by name, so Neo4j is never
    # asked for the relationships of Interstellar.
    assert metadata["truncated_steps"] == [0]
    assert metadata["path"][0].endswith("{Inception, Interstellar}")
    assert [item.content for item in result.items] == [
        "Sci-Fi (:Genre) | path: Christopher Nolan <-[directed_by]- Inception "
        "-[has_genre]-> Sci-Fi"
    ]


def test_decision_walk_several_start_nodes(make_retriever: Callable[..., Any]) -> None:
    retriever, _ = make_retriever(["follow directed_by", "STOP"])
    result = retriever.search(
        query_text="q", start_node_ids=["interstellar", "inception"]
    )
    (state, options, _) = calls_of(retriever)[0]
    assert state["current_entities"] == ["Inception", "Interstellar"]
    assert options["0"] == "STOP: the answer is: Inception, Interstellar"
    # Nolan is reached from both films, and the path shows the film that comes first by name.
    assert [item.metadata["path"] for item in result.items if item.metadata] == [
        "Inception -[directed_by]-> Christopher Nolan"
    ]


def test_decision_walk_formats_unlabelled_node(
    make_retriever: Callable[..., Any],
) -> None:
    retriever, _ = make_retriever([])
    item = retriever.default_record_formatter(
        neo4j.Record(
            {"node_id": "x", "name": "x", "labels": [], "properties": {}, "path": "x"}
        )
    )
    assert item.content == "x | path: x"


def test_decision_walk_limits_result_nodes(make_retriever: Callable[..., Any]) -> None:
    retriever, _ = make_retriever(
        ["follow directed_by", "follow directed_by"], max_result_nodes=1
    )
    result = retriever.search(query_text="q", start_node_ids=["nolan"], max_steps=1)
    assert [item.metadata["node_id"] for item in result.items if item.metadata] == [
        "inception"
    ]
    assert (result.metadata or {})["result_nodes_total"] == 2


def test_decision_walk_long_names_are_shortened(
    make_retriever: Callable[..., Any],
) -> None:
    retriever, _ = make_retriever(["STOP"], max_names_in_option=1)
    retriever.search(query_text="q", start_node_ids=["nolan"])
    (_, options, _) = calls_of(retriever)[0]
    assert options["1"] == "follow directed_by (incoming) to: Inception ... (+1 more)"


def test_decision_walk_names_in_state_are_capped() -> None:
    names = {str(i): f"n{i}" for i in range(105)}
    shown = DecisionWalkRetriever._names_in_state(list(names), names)
    assert len(shown) == 101
    assert shown[-1] == "... (+5 more)"


def test_decision_walk_custom_result_formatter(
    make_retriever: Callable[..., Any],
) -> None:
    def formatter(record: neo4j.Record) -> RetrieverResultItem:
        return RetrieverResultItem(content=record["name"])

    retriever, _ = make_retriever(["STOP"], result_formatter=formatter)
    result = retriever.search(query_text="q", start_node_ids=["inception"])
    assert [item.content for item in result.items] == ["Inception"]


def test_decision_walk_invalid_choice(make_retriever: Callable[..., Any]) -> None:
    retriever, _ = make_retriever([])
    retriever.decision_model.decide = MagicMock(
        return_value=DecisionResponse(choice="9", probabilities={"9": 1.0})
    )
    with pytest.raises(DecisionModelError, match="not one of the options"):
        retriever.search(query_text="q", start_node_ids=["inception"])


# ---- tournament -------------------------------------------------------------------


def test_decision_walk_tournament(make_retriever: Callable[..., Any]) -> None:
    # Inception has STOP and 3 other options, and the model accepts 2 options at a time.
    # The first round has the buckets [1, 2] and [3]. Option 3 goes through without a
    # call, then the model decides between [1, 3], and the winner is compared with STOP.
    retriever, _ = make_retriever(
        ["follow directed_by", "follow starred_actors", "STOP"],
        max_options=2,
    )
    result = retriever.search(query_text="q", start_node_ids=["inception"], max_steps=1)
    calls = calls_of(retriever)
    assert [list(options.values()) for _, options, _ in calls] == [
        [
            "follow directed_by (outgoing) to: Christopher Nolan",
            "follow has_genre (outgoing) to: Sci-Fi",
        ],
        [
            "follow directed_by (outgoing) to: Christopher Nolan",
            "follow starred_actors (outgoing) to: Leonardo DiCaprio",
        ],
        [
            "STOP: the answer is: Inception",
            "follow starred_actors (outgoing) to: Leonardo DiCaprio",
        ],
    ]
    assert (result.metadata or {})["stopped_by"] == "model_stop"
    assert [d["option_indexes"] for d in (result.metadata or {})["decisions"]] == [
        [1, 2],
        [1, 3],
        [0, 3],
    ]


def test_split_into_buckets() -> None:
    assert [len(b) for b in split_into_buckets(list(range(280)), 255)] == [140, 140]
    assert [len(b) for b in split_into_buckets(list(range(135)), 26)] == [
        23,
        23,
        23,
        22,
        22,
        22,
    ]
    assert split_into_buckets([1, 2, 3], 2) == [[1, 2], [3]]


# ---- start node -------------------------------------------------------------------


def test_decision_walk_finds_start_with_fulltext(
    make_retriever: Callable[..., Any],
) -> None:
    retriever, graph = make_retriever(
        ["start at Interstellar", "STOP"],
        seed_hits=["inception", "interstellar"],
        fulltext_index_name="names",
        seed_top_k=5,
    )
    result = retriever.search(query_text="[Interstellar] genre?")

    query, parameters = graph.queries[0]
    assert query == SEED_QUERY
    assert parameters["index_name"] == "names"
    assert parameters["query_text"] == "\\[Interstellar\\] genre\\?"
    assert parameters["top_k"] == 5
    state, options, instructions = calls_of(retriever)[0]
    assert state == {"question": "[Interstellar] genre?"}
    assert options == {
        "0": "start at Inception (Movie)",
        "1": "start at Interstellar (Movie)",
    }
    assert instructions == retriever.seed_instructions
    metadata = result.metadata or {}
    assert metadata["start_nodes"] == ["Interstellar"]
    assert [d["kind"] for d in metadata["decisions"]] == ["start", "step"]


def test_decision_walk_single_candidate_needs_no_decision(
    make_retriever: Callable[..., Any],
) -> None:
    retriever, _ = make_retriever(
        ["STOP"], seed_hits=["titanic"], fulltext_index_name="names"
    )
    result = retriever.search(query_text="Titanic")
    assert (result.metadata or {})["start_nodes"] == ["Titanic"]
    assert len(calls_of(retriever)) == 1


@pytest.mark.parametrize(
    "query_text, seed_hits", [("nothing matches", []), ("  ", ["titanic"])]
)
def test_decision_walk_no_start_candidate(
    make_retriever: Callable[..., Any], query_text: str, seed_hits: list[str]
) -> None:
    retriever, _ = make_retriever([], seed_hits=seed_hits, fulltext_index_name="names")
    result = retriever.search(query_text=query_text)
    assert result.items == []
    assert (result.metadata or {})["stopped_by"] == "no_start_node"


def test_decision_walk_needs_a_way_to_start(make_retriever: Callable[..., Any]) -> None:
    retriever, _ = make_retriever([])
    with pytest.raises(DecisionWalkRetrievalError, match="fulltext_index_name"):
        retriever.search(query_text="q")


def test_escape_lucene() -> None:
    assert (
        escape_lucene('[Tom Hanks] AND "Big"?  NOTE OR')
        == '\\[Tom Hanks\\] and \\"Big\\"\\? NOTE or'
    )
    assert escape_lucene("a+b-c:d/e\\f") == "a\\+b\\-c\\:d\\/e\\\\f"
    assert escape_lucene("word " * 300) == " ".join(["word"] * 256)
