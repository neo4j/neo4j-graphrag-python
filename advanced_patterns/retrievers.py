from typing import List, Dict, Any
from neo4j_graphrag.retrievers import VectorRetriever

class HybridRetriever(VectorRetriever):
    """
    A custom retriever that combines vector similarity, full-text search,
    and graph traversal for more accurate results.
    """
    def __init__(self, neo4j_driver, embedding_model):
        super().__init__(
            neo4j_driver,
            embedding_model=embedding_model,
            node_label="Chunk",
            embedding_property="embedding",
            text_property="text"
        )
    
    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        # Generate query embedding
        query_embedding = self.embedding_model.embed_query(query)
        
        # Cypher query combining multiple retrieval strategies
        cypher_query = """
        CALL db.index.vector.queryNodes('chunk_embeddings', $k, $embedding)
        YIELD node as chunk, score as vector_score
        WHERE chunk:Chunk
        
        // Full-text search on chunk content
        CALL db.index.fulltext.queryNodes('chunk_text', $query)
        YIELD node as text_chunk, score as text_score
        WHERE text_chunk = chunk
        
        // Graph traversal to related entities
        MATCH (chunk)-[r]->(entity:__Entity__)
        WHERE entity.type IN ['Disease', 'Symptom', 'Treatment', 'Biomarker']
        
        // Combine scores and return results
        RETURN 
            chunk,
            collect(distinct entity) as entities,
            vector_score * 0.6 + text_score * 0.4 as combined_score
        ORDER BY combined_score DESC
        LIMIT 5
        """
        
        # Execute query with parameters
        results = self.neo4j_driver.execute_query(
            cypher_query,
            embedding=query_embedding,
            query=query,
            k=10
        )
        
        return [record.data() for record in results]
