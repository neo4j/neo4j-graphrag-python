import os
import neo4j
import pygraphviz as pgv
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.retrievers import VectorRetriever

# Check for OpenAI API key
if "OPENAI_API_KEY" not in os.environ:
    raise ValueError("Please set the OPENAI_API_KEY environment variable")

# Connect to Neo4j
neo4j_driver = neo4j.GraphDatabase.driver(
    "neo4j://localhost:7687",
    auth=("neo4j", "password")
)

# Initialize embeddings model
embedder = OpenAIEmbeddings()

def visualize_knowledge_subgraph(query, output_file="knowledge_graph.svg", k=3):
    """
    Create a visualization of a knowledge subgraph based on a query.
    
    Args:
        query (str): The query to retrieve relevant chunks
        output_file (str): Path to save the visualization
        k (int): Number of similar chunks to retrieve
    """
    # Create a graph visualization
    G = pgv.AGraph(strict=False, directed=True)
    
    # Set up the retriever
    retriever = VectorRetriever(
        neo4j_driver,
        embedding_model=embedder,
        node_label="Chunk",
        embedding_property="embedding",
        text_property="text",
        k=k
    )
    
    # Query for nodes and relationships
    results = retriever.retrieve(query)
    
    # Add nodes and edges to the visualization
    for record in results:
        # Add chunk node
        chunk_text = record["chunk"].text[:50] + "..."
        G.add_node(record["chunk"].id, 
                  label=chunk_text,
                  shape="box",
                  style="rounded,filled",
                  fillcolor="#E8F5E9")
        
        # Add entity nodes and relationships
        for entity in record.get("entities", []):
            # Add entity node if it doesn't exist
            if not G.has_node(entity.id):
                G.add_node(entity.id,
                          label=f"{entity.type}\n{entity.text[:30]}...",
                          shape="ellipse",
                          style="filled",
                          fillcolor="#E3F2FD")
            
            # Add relationship
            G.add_edge(record["chunk"].id,
                      entity.id,
                      color="blue",
                      penwidth=1.5)
    
    # Set graph-wide attributes
    G.graph_attr.update({
        "rankdir": "LR",
        "splines": "ortho",
        "nodesep": "0.5",
        "ranksep": "1.0",
        "bgcolor": "white"
    })
    
    # Generate the visualization
    G.layout(prog="dot")
    G.draw(output_file)
    print(f"Graph visualization saved to {output_file}")

if __name__ == "__main__":
    # Example queries to visualize different aspects of the knowledge graph
    queries = [
        "What are the common biomarkers for SLE?",
        "What treatments are available for lupus?",
        "What are the main symptoms of SLE?"
    ]
    
    # Create visualizations for each query
    for i, query in enumerate(queries):
        output_file = f"knowledge_graph_{i+1}.svg"
        print(f"\nVisualizing knowledge graph for query: {query}")
        visualize_knowledge_subgraph(query, output_file)
