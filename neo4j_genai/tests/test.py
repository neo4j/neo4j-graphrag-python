from neo4j_genai import Client
from neo4j_genai import GenAI
from neo4j_genai import GenAIClient
from neo4j_genai import VectorClient

from neo4j import GraphDatabase, Driver
from typing import Optional
from langchain_core.embeddings import Embeddings

from pydantic_v1 import BaseModel

URI = "neo4j://localhost:7687"
AUTH = ("neo4j", "password")

driver = GraphDatabase.driver(URI, auth=AUTH)

class GenAIClient:
    def __init__(self, driver: Driver, embeddings: Optional[]) -> None:
        pass
client = GenAIClient(driver, embeddings=)

client.create_vector_index()
client.drop_vector_index()

"""**Embeddings** interface."""
from abc import ABC, abstractmethod
from typing import List


class Embeddings(ABC):
    """Interface for embedding models."""
    
    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
