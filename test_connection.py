import os
import neo4j
from neo4j_graphrag.llm import OpenAILLM as LLM
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings as Embeddings

# Set OpenAI API Key - should be set in environment or .env file
if "OPENAI_API_KEY" not in os.environ:
    raise ValueError("Please set the OPENAI_API_KEY environment variable")

# Test OpenAI connection
llm = LLM(model_name="gpt-4")
embedder = Embeddings()

print("Testing OpenAI connection...")
try:
    # Test embeddings
    test_embedding = embedder.embed_query("Test query")
    print("✓ OpenAI embeddings working")
except Exception as e:
    print("✗ OpenAI embeddings error:", str(e))

# Connect to Neo4j
print("\nTesting Neo4j connection...")
try:
    neo4j_driver = neo4j.GraphDatabase.driver(
        "neo4j://localhost:7687",
        auth=("neo4j", "password")
    )

    # Test the connection
    with neo4j_driver.session() as session:
        result = session.run("RETURN 1 as num")
        print("✓ Neo4j connection working:", result.single()["num"])

    neo4j_driver.close()
except Exception as e:
    print("✗ Neo4j connection error:", str(e))
