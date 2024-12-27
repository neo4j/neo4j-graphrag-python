import os
import asyncio
from pathlib import Path
from neo4j_graphrag.experimental.pipeline.config.runner import PipelineRunner

# Custom prompt for medical entity extraction
medical_prompt = '''
You are a medical researcher tasked with extracting information from papers 
and structuring it in a property graph to inform further medical and research Q&A.

Extract the entities (nodes) and specify their type from the following Input text.
Also extract the relationships between these nodes. The relationship direction goes from the start node to the end node.

Guidelines:
1. Focus on medical entities like diseases, symptoms, biomarkers, treatments, and biological processes
2. Extract specific values, measurements, and ranges when available
3. Preserve hierarchical relationships (e.g., a protein being part of a pathway)
4. Note temporal relationships and sequences of events
5. Capture research findings and their evidence levels

Return result as a JSON object with the following structure:
{
    "nodes": [
        {
            "id": string,  # unique identifier
            "type": string,  # one of the allowed node types
            "properties": {  # additional properties
                "name": string,
                "value": string or number (optional),
                "unit": string (optional),
                ...
            }
        }
    ],
    "relationships": [
        {
            "from": string,  # source node id
            "to": string,    # target node id
            "type": string,  # one of the allowed relationship types
            "properties": {  # additional properties
                ...
            }
        }
    ]
}

Input text: {text}
'''

async def build_knowledge_graph(file_path: str):
    """Build knowledge graph from a PDF file"""
    # Check for environment variables
    required_vars = ['NEO4J_URI', 'NEO4J_USERNAME', 'NEO4J_PASSWORD', 'OPENAI_API_KEY']
    missing_vars = [var for var in required_vars if var not in os.environ]
    
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    # Create pipeline runner from config file
    config_path = Path(__file__).parent / "medical_kg_pipeline_config.json"
    runner = PipelineRunner.from_config_file(config_path)
    
    # Build knowledge graph
    try:
        await runner.run({"file_path": file_path})
        print(f"Successfully processed {file_path}")
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
    finally:
        await runner.close()

async def main():
    # Process all PDF files in the data/papers directory
    papers_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "papers")
    
    for file_name in os.listdir(papers_dir):
        if file_name.endswith('.pdf'):
            file_path = os.path.join(papers_dir, file_name)
            print(f"\nProcessing {file_name}...")
            await build_knowledge_graph(file_path)

if __name__ == "__main__":
    asyncio.run(main())
