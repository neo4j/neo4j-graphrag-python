"""GraphRAG with optional explainability output.

Runs against the public Neo4j recommendations demo database and compares a
standard GraphRAG answer with structured provenance (sources, graph paths).

Requires OPENAI_API_KEY in the environment and the ``openai`` optional dependency::

    uv sync --extra openai

Examples::

    uv run examples/question_answering/graphrag_with_explain.py
    uv run examples/question_answering/graphrag_with_explain.py --no-explain
    uv run examples/question_answering/graphrag_with_explain.py --format json
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import neo4j
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.generation import (
    ExplainConfig,
    ExplainResult,
    GraphRAG,
    MOVIES_ACTORS_PATH_RETRIEVAL_QUERY,
    movies_vector_cypher_explain_formatter,
)
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.retrievers import VectorCypherRetriever

URI = "neo4j+s://demo.neo4jlabs.com"
AUTH = ("recommendations", "recommendations")
DATABASE = "recommendations"
INDEX = "moviePlotsEmbedding"
DEFAULT_QUESTION = "Who were the actors in Avatar?"
OPENAI_EXTRA_INSTALL_HINT = (
    "This example requires the openai optional dependency.\n"
    "Install it with: uv sync --extra openai"
)


def ensure_openai_extra() -> None:
    try:
        import openai  # noqa: F401
    except ImportError as exc:
        raise SystemExit(OPENAI_EXTRA_INSTALL_HINT) from exc


def build_rag(driver: neo4j.Driver) -> GraphRAG:
    retriever = VectorCypherRetriever(
        driver,
        index_name=INDEX,
        retrieval_query=MOVIES_ACTORS_PATH_RETRIEVAL_QUERY,
        result_formatter=movies_vector_cypher_explain_formatter,
        embedder=OpenAIEmbeddings(),
        neo4j_database=DATABASE,
    )
    llm = OpenAILLM(model_name="gpt-4o-mini", model_params={"temperature": 0})
    return GraphRAG(retriever=retriever, llm=llm)


def _node_label(node: dict[str, Any]) -> str:
    labels = node.get("labels") or []
    name = (node.get("properties") or {}).get("name") or (
        node.get("properties") or {}
    ).get("title")
    if labels and name:
        return f"{labels[0]}({name})"
    if labels:
        return labels[0]
    if name:
        return str(name)
    return "node"


def _format_path(path: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for element in path:
        if "type" in element:
            parts.append(f"-[:{element['type']}]->")
        else:
            parts.append(_node_label(element))
    return " ".join(parts)


def format_explain_table(explain: ExplainResult) -> str:
    lines = [f"Retriever: {explain.trace.retriever}", "", "Sources:"]
    for source in explain.sources:
        score = f" (score={source.score:.3f})" if source.score is not None else ""
        lines.append(f"  [{source.index}]{score} {source.content}")

    if explain.graph:
        lines.extend(["", "Graph context:"])
        for index, context in enumerate(explain.graph, start=1):
            lines.append(f"  Source {index}:")
            if context.seed_node is not None:
                lines.append(f"    seed: {_node_label(context.seed_node.model_dump())}")
            for path_index, path in enumerate(context.paths, start=1):
                serialized_path = [element.model_dump() for element in path]
                lines.append(f"    path {path_index}: {_format_path(serialized_path)}")
    return "\n".join(lines)


def result_to_json(answer: str, explain: ExplainResult | None) -> dict[str, Any]:
    payload: dict[str, Any] = {"answer": answer}
    if explain is not None:
        payload["explain"] = explain.model_dump(mode="json")
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "question",
        nargs="?",
        default=DEFAULT_QUESTION,
        help=f"Question to ask (default: {DEFAULT_QUESTION!r})",
    )
    parser.add_argument(
        "--explain",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Attach structured provenance to the GraphRAG result (default: on)",
    )
    parser.add_argument(
        "--format",
        choices=("table", "json"),
        default="table",
        help="Output format (default: table)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of retrieval results to pass to the LLM (default: 3)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ensure_openai_extra()
    with neo4j.GraphDatabase.driver(URI, auth=AUTH) as driver:
        rag = build_rag(driver)
        result = rag.search(
            args.question,
            explain=ExplainConfig() if args.explain else None,
            retriever_config={"top_k": args.top_k},
        )

    if args.format == "json":
        print(json.dumps(result_to_json(result.answer, result.explain), indent=2))
        return 0

    print(f"Question: {args.question}")
    print()
    print("Answer:")
    print(result.answer)
    if result.explain is not None:
        print()
        print(format_explain_table(result.explain))
    return 0


if __name__ == "__main__":
    sys.exit(main())
