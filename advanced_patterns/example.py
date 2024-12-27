import os
import neo4j
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.generation import GraphRAG

from retrievers import HybridRetriever

# Check for OpenAI API key
if "OPENAI_API_KEY" not in os.environ:
    raise ValueError("Please set the OPENAI_API_KEY environment variable")

# Connect to Neo4j
neo4j_driver = neo4j.GraphDatabase.driver(
    "neo4j://localhost:7687",
    auth=("neo4j", "password")
)

# Initialize components
llm = OpenAILLM(model_name="gpt-4")
embedder = OpenAIEmbeddings()

# Create custom prompt template
medical_prompt = """You are a medical research assistant analyzing information about Systemic Lupus Erythematosus (SLE).
Use ONLY the provided context to answer questions. If the information is not in the context, say "I don't have enough information to answer that."

For treatments and medications:
- Specify the purpose and typical usage
- Note any mentioned side effects or contraindications
- Indicate if it's a first-line or alternative treatment

For symptoms and biomarkers:
- List them in order of frequency/importance
- Note any correlations with specific organ involvement
- Mention typical values or ranges if provided

Context: {context}

Question: {query}

Answer:"""

def demo_hybrid_retriever():
    """Demonstrate the HybridRetriever with custom prompt"""
    print("\n=== Hybrid Retriever Demo ===")
    
    # Create hybrid retriever
    retriever = HybridRetriever(neo4j_driver, embedder)
    
    # Create GraphRAG with custom prompt
    rag = GraphRAG(
        llm=llm,
        retriever=retriever,
        prompt_template=medical_prompt
    )
    
    # Example queries
    queries = [
        "What are the main biomarkers used to diagnose and monitor SLE?",
        "What treatments are available for severe SLE symptoms?",
        "How do different biomarkers correlate with organ involvement in SLE?"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        response = rag.search(query)
        print(f"Response: {response}")

if __name__ == "__main__":
    try:
        # Run demonstration
        demo_hybrid_retriever()
    finally:
        # Clean up
        neo4j_driver.close()
