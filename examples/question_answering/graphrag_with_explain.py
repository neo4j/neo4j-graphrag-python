"""GraphRAG with optional explainability output.

Demonstrates explain on the public Neo4j recommendations demo database with either:

- **text2cypher** (default): natural-language questions translated to Cypher; shows
  generated query, sources, and graph paths.
- **vector-cypher**: plot-similarity search plus graph expansion; shows similarity
  scores, sources, and actor/director paths for each hit.

Requires OPENAI_API_KEY in the environment and the ``openai`` optional dependency::

    uv sync --extra openai

Examples::

    uv run examples/question_answering/graphrag_with_explain.py
    uv run examples/question_answering/graphrag_with_explain.py --no-explain
    uv run examples/question_answering/graphrag_with_explain.py --format json \\
        "Which movies connect Tom Hanks and Kevin Bacon through Ron Howard?"
    uv run examples/question_answering/graphrag_with_explain.py --retriever vector-cypher \\
        "Find movies about a marine on an alien planet and list the cast and director."
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Literal

import neo4j
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.generation import (
    ExplainConfig,
    ExplainResult,
    GraphRAG,
    text2cypher_explain_result_formatter,
)
from neo4j_graphrag.generation.explain_recommendations import (
    MOVIES_ACTORS_PATH_RETRIEVAL_QUERY,
    movies_vector_cypher_explain_formatter,
)
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.retrievers import Text2CypherRetriever, VectorCypherRetriever

URI = "neo4j+s://demo.neo4jlabs.com"
AUTH = ("recommendations", "recommendations")
DATABASE = "recommendations"
INDEX_NAME = "moviePlotsEmbedding"
RetrieverName = Literal["text2cypher", "vector-cypher"]
DEFAULT_QUESTIONS: dict[RetrieverName, str] = {
    "text2cypher": (
        "Which movies connect Tom Hanks and Kevin Bacon through Ron Howard as director?"
    ),
    "vector-cypher": (
        "Find movies about a marine on an alien planet and list the cast and director."
    ),
}
OPENAI_EXTRA_INSTALL_HINT = (
    "This example requires the openai optional dependency.\n"
    "Install it with: uv sync --extra openai"
)

RECOMMENDATIONS_NEO4J_SCHEMA = """
Node properties:
Actor {name: STRING}
Director {name: STRING}
Person {name: STRING, born: INTEGER}
User {name: STRING, userId: STRING}
Genre {name: STRING}
Movie {tagline: STRING, title: STRING, released: INTEGER}
Relationship properties:
ACTED_IN {roles: LIST}
DIRECTED {}
RATED {rating: FLOAT, timestamp: INTEGER}
The relationships:
(:Actor)-[:ACTED_IN]->(:Movie)
(:Person)-[:ACTED_IN]->(:Movie)
(:Person)-[:DIRECTED]->(:Movie)
(:Director)-[:DIRECTED]->(:Movie)
(:User)-[:RATED]->(:Movie)
(:Movie)-[:IN_GENRE]->(:Genre)
""".strip()

RECOMMENDATIONS_TEXT2CYPHER_EXAMPLES = [
    (
        "USER INPUT: 'Which movies did Joel Coen and Steve Buscemi work on together?' "
        "QUERY: MATCH path = (d:Person)-[:DIRECTED]->(m:Movie)<-[:ACTED_IN]-(a:Actor) "
        "WHERE d.name = 'Joel Coen' AND a.name = 'Steve Buscemi' "
        "RETURN m.title AS title, path AS path"
    ),
    (
        "USER INPUT: 'Which movies connect Tom Hanks and Kevin Bacon through Ron Howard?' "
        "QUERY: MATCH path = (hanks:Actor)-[:ACTED_IN]->(m1:Movie)<-[:DIRECTED]-"
        "(d:Person)-[:DIRECTED]->(m2:Movie)<-[:ACTED_IN]-(bacon:Actor) "
        "WHERE hanks.name = 'Tom Hanks' AND d.name = 'Ron Howard' "
        "AND bacon.name = 'Kevin Bacon' "
        "RETURN m1.title AS hanks_movie, m2.title AS bacon_movie, path AS path"
    ),
    (
        "USER INPUT: 'Which other movies in the same genre as Avatar did Sam Worthington "
        "also act in?' "
        "QUERY: MATCH path = (a:Actor)-[:ACTED_IN]->(avatar:Movie)-[:IN_GENRE]->(g:Genre)"
        "<-[:IN_GENRE]-(other:Movie)<-[:ACTED_IN]-(a) "
        "WHERE a.name = 'Sam Worthington' AND avatar.title = 'Avatar' "
        "AND other.title <> avatar.title "
        "RETURN other.title AS movie, g.name AS genre, path AS path"
    ),
    (
        "USER INPUT: 'Who co-starred with Keanu Reeves in The Matrix?' "
        "QUERY: MATCH path = (k:Actor)-[:ACTED_IN]->(m:Movie)<-[:ACTED_IN]-(co:Actor) "
        "WHERE k.name = 'Keanu Reeves' AND m.title = 'Matrix, The' "
        "AND co.name <> k.name "
        "RETURN co.name AS costar, path AS path"
    ),
]


def ensure_openai_extra() -> None:
    try:
        import openai  # noqa: F401
    except ImportError as exc:
        raise SystemExit(OPENAI_EXTRA_INSTALL_HINT) from exc


def build_text2cypher_rag(driver: neo4j.Driver, llm: OpenAILLM) -> GraphRAG:
    retriever = Text2CypherRetriever(
        driver,
        llm=llm,
        neo4j_schema=RECOMMENDATIONS_NEO4J_SCHEMA,
        examples=RECOMMENDATIONS_TEXT2CYPHER_EXAMPLES,
        result_formatter=text2cypher_explain_result_formatter,
        neo4j_database=DATABASE,
    )
    return GraphRAG(retriever=retriever, llm=llm)


def build_vector_cypher_rag(
    driver: neo4j.Driver,
    llm: OpenAILLM,
    embedder: OpenAIEmbeddings,
) -> GraphRAG:
    retriever = VectorCypherRetriever(
        driver,
        index_name=INDEX_NAME,
        retrieval_query=MOVIES_ACTORS_PATH_RETRIEVAL_QUERY,
        result_formatter=movies_vector_cypher_explain_formatter,
        embedder=embedder,
        neo4j_database=DATABASE,
    )
    return GraphRAG(retriever=retriever, llm=llm)


def build_rag(
    driver: neo4j.Driver,
    llm: OpenAILLM,
    *,
    retriever: RetrieverName,
    embedder: OpenAIEmbeddings | None = None,
) -> GraphRAG:
    if retriever == "vector-cypher":
        if embedder is None:
            embedder = OpenAIEmbeddings()
        return build_vector_cypher_rag(driver, llm, embedder)
    return build_text2cypher_rag(driver, llm)


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
        "--retriever",
        choices=("text2cypher", "vector-cypher"),
        default="text2cypher",
        help="Retriever to use (default: text2cypher)",
    )
    parser.add_argument(
        "question",
        nargs="?",
        default=None,
        help="Question to ask (default depends on --retriever)",
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
        help="Number of vector hits for vector-cypher mode (default: 3)",
    )
    args = parser.parse_args(argv)
    retriever_name: RetrieverName = args.retriever
    if args.question is None:
        args.question = DEFAULT_QUESTIONS[retriever_name]
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ensure_openai_extra()
    llm = OpenAILLM(model_name="gpt-4o-mini", model_params={"temperature": 0})
    embedder = OpenAIEmbeddings() if args.retriever == "vector-cypher" else None
    retriever_config = (
        {"top_k": args.top_k} if args.retriever == "vector-cypher" else {}
    )
    with neo4j.GraphDatabase.driver(URI, auth=AUTH) as driver:
        rag = build_rag(
            driver,
            llm,
            retriever=args.retriever,
            embedder=embedder,
        )
        result = rag.search(
            args.question,
            explain=ExplainConfig() if args.explain else None,
            retriever_config=retriever_config,
        )

    if args.format == "json":
        print(json.dumps(result_to_json(result.answer, result.explain), indent=2))
        return 0

    print(f"Retriever: {args.retriever}")
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
