"""The example uses the DecisionWalkRetriever to fetch context.
A decision model walks the graph one step at a time, and it chooses which
relationship to follow and when to stop. The example uses the OpenRouterDecisionModel,
so the OPENROUTER_API_KEY environment variable must be set.
"""

import neo4j
from neo4j_graphrag.decision import OpenRouterDecisionModel
from neo4j_graphrag.retrievers import DecisionWalkRetriever

# Define database credentials
URI = "neo4j+s://demo.neo4jlabs.com"
AUTH = ("recommendations", "recommendations")
DATABASE = "recommendations"

# Create the decision model, pinned to one OpenRouter provider
decision_model = OpenRouterDecisionModel(
    "liquid/d1-ultra",
    provider={"order": ["liquid"], "allow_fallbacks": False},
)

with neo4j.GraphDatabase.driver(URI, auth=AUTH) as driver:
    # Initialize the retriever. The start node is found with an existing
    # full-text index on Person names, then chosen by the decision model.
    retriever = DecisionWalkRetriever(
        driver=driver,
        decision_model=decision_model,
        fulltext_index_name="personFulltext",
        node_name_properties=["name", "title"],
        neo4j_database=DATABASE,
    )

    # Walk the graph, one decision at a time, to the nodes that answer the question
    query_text = "Which movies did Hugo Weaving act in?"
    result = retriever.search(query_text=query_text)
    for item in result.items:
        print(item.content)
    print(result.metadata["path"] if result.metadata else None)

decision_model.close()
