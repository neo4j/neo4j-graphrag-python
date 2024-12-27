import os
import neo4j
from neo4j_graphrag.llm import OpenAILLM as LLM
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings as Embeddings
from neo4j_graphrag.retrievers import VectorRetriever

# Set OpenAI API Key - should be set in environment or .env file
if "OPENAI_API_KEY" not in os.environ:
    raise ValueError("Please set the OPENAI_API_KEY environment variable")

# Connect to Neo4j
neo4j_driver = neo4j.GraphDatabase.driver(
    "neo4j://localhost:7687",
    auth=("neo4j", "password")
)

# Initialize components
llm = LLM(model_name="gpt-4")
embedder = Embeddings()

# Create vector retriever
vector_retriever = VectorRetriever(
    neo4j_driver,
    index_name="text_embeddings",
    embedder=embedder
)

# Test a simple query
response = vector_retriever.search("What happens in the Chamber of Secrets?")
print("\nQuery Results:")
for doc in response:
    print(f"\nScore: {doc.score}")
    print(f"Content: {doc.content}")
