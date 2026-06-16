"""GraphRAG with optional explainability output.

Uses Text2Cypher against the public Neo4j recommendations demo database so you
can ask natural-language questions about the graph and inspect sources plus the
generated Cypher query.

Requires OPENAI_API_KEY in the environment and the ``openai`` optional dependency::

    uv sync --extra openai

Examples::

    uv run examples/question_answering/graphrag_with_explain.py
    uv run examples/question_answering/graphrag_with_explain.py --no-explain
    uv run examples/question_answering/graphrag_with_explain.py --format json \\
        "Which movies did Joel Coen and Steve Buscemi work on together?"
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import neo4j
from neo4j_graphrag.generation import (
    ExplainConfig,
    ExplainResult,
    GraphRAG,
    text2cypher_explain_result_formatter,
)
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.retrievers import Text2CypherRetriever

URI = "neo4j+s://demo.neo4jlabs.com"
AUTH = ("recommendations", "recommendations")
DATABASE = "recommendations"
DEFAULT_QUESTION = "Who were the actors in Avatar?"
OPENAI_EXTRA_INSTALL_HINT = (
    "This example requires the openai optional dependency.\n"
    "Install it with: uv sync --extra openai"
)

RECOMMENDATIONS_NEO4J_SCHEMA = """
Node properties:
Actor {name: STRING}
Director {name: STRING}
Person {name: STRING, born: INTEGER}
Movie {tagline: STRING, title: STRING, released: INTEGER}
Relationship properties:
ACTED_IN {roles: LIST}
DIRECTED {}
REVIEWED {summary: STRING, rating: INTEGER}
The relationships:
(:Actor)-[:ACTED_IN]->(:Movie)
(:Person)-[:ACTED_IN]->(:Movie)
(:Person)-[:DIRECTED]->(:Movie)
(:Director)-[:DIRECTED]->(:Movie)
(:Person)-[:REVIEWED]->(:Movie)
""".strip()

RECOMMENDATIONS_TEXT2CYPHER_EXAMPLES = [
    (
        "USER INPUT: 'Which actors starred in the Matrix?' "
        "QUERY: MATCH path = (p:Person)-[:ACTED_IN]->(m:Movie) "
        "WHERE m.title = 'The Matrix' "
        "RETURN p.name AS name, path AS path"
    ),
    (
        "USER INPUT: 'Who directed One Flew Over the Cuckoo\\'s Nest?' "
        "QUERY: MATCH path = (p:Person)-[:DIRECTED]->(m:Movie) "
        'WHERE m.title = "One Flew Over the Cuckoo\'s Nest" '
        "RETURN p.name AS name, path AS path"
    ),
    (
        "USER INPUT: 'Which movies did Joel Coen and Steve Buscemi work on together?' "
        "QUERY: MATCH path = (d:Person)-[:DIRECTED]->(m:Movie)<-[:ACTED_IN]-(a:Actor) "
        "WHERE d.name = 'Joel Coen' AND a.name = 'Steve Buscemi' "
        "RETURN m.title AS title, path AS path"
    ),
]


def ensure_openai_extra() -> None:
    try:
        import openai  # noqa: F401
    except ImportError as exc:
        raise SystemExit(OPENAI_EXTRA_INSTALL_HINT) from exc


def build_rag(driver: neo4j.Driver, llm: OpenAILLM) -> GraphRAG:
    retriever = Text2CypherRetriever(
        driver,
        llm=llm,
        neo4j_schema=RECOMMENDATIONS_NEO4J_SCHEMA,
        examples=RECOMMENDATIONS_TEXT2CYPHER_EXAMPLES,
        result_formatter=text2cypher_explain_result_formatter,
        neo4j_database=DATABASE,
    )
    return GraphRAG(retriever=retriever, llm=llm)


def format_explain_table(explain: ExplainResult) -> str:
    lines = [f"Retriever: {explain.trace.retriever}"]
    if explain.trace.cypher:
        lines.extend(["", "Generated Cypher:", f"  {explain.trace.cypher}"])
    lines.extend(["", "Sources:"])
    if not explain.sources:
        lines.append("  (no rows returned)")
    for source in explain.sources:
        score = f" (score={source.score:.3f})" if source.score is not None else ""
        lines.append(f"  [{source.index}]{score} {source.content}")

    if explain.graph:
        lines.extend(["", "Graph context:"])
        for index, context in enumerate(explain.graph, start=1):
            lines.append(f"  Source {index}:")
            if context.seed_node is not None:
                seed = context.seed_node
                label = seed.labels[0] if seed.labels else "node"
                name = seed.properties.get("title") or seed.properties.get("name")
                seed_text = f"{label}({name})" if name else label
                lines.append(f"    seed: {seed_text}")
            for path_index, path in enumerate(context.paths, start=1):
                parts: list[str] = []
                for element in path:
                    if hasattr(element, "type"):
                        parts.append(f"-[:{element.type}]->")
                    else:
                        label = element.labels[0] if element.labels else "node"
                        name = element.properties.get("name") or element.properties.get(
                            "title"
                        )
                        parts.append(f"{label}({name})" if name else label)
                lines.append(f"    path {path_index}: {' '.join(parts)}")
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ensure_openai_extra()
    llm = OpenAILLM(model_name="gpt-4o-mini", model_params={"temperature": 0})
    with neo4j.GraphDatabase.driver(URI, auth=AUTH) as driver:
        rag = build_rag(driver, llm)
        result = rag.search(
            args.question,
            explain=ExplainConfig() if args.explain else None,
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
