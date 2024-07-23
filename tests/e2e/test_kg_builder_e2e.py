import neo4j
import pytest
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from neo4j_genai.generation.kg_writer import Neo4jWriter
from neo4j_genai.generation.types import (
    Neo4jEmbeddingProperty,
    Neo4jGraph,
    Neo4jNode,
    Neo4jProperty,
    Neo4jRelationship,
)

pytest_plugins = ("pytest_asyncio",)


@pytest.mark.asyncio
@pytest.mark.usefixtures("setup_neo4j_for_kg_builder")
async def test_kg_writer(driver: neo4j.Driver) -> None:
    start_node = Neo4jNode(
        id="1",
        label="Document",
        properties=[Neo4jProperty(key="chunk", value=1)],
        embedding_properties=[
            Neo4jEmbeddingProperty(
                key="vectorProperty", value="Lorem ipsum dolor sit amet."
            )
        ],
    )
    end_node = Neo4jNode(
        id="2",
        label="Document",
        properties=[Neo4jProperty(key="chunk", value=2)],
        embedding_properties=[
            Neo4jEmbeddingProperty(
                key="vectorProperty",
                value="Nulla facilisi. Pellentesque habitant morbi.",
            )
        ],
    )
    relationship = Neo4jRelationship(
        start_node_id="1", end_node_id="2", label="NEXT_CHUNK"
    )
    graph = Neo4jGraph(nodes=[start_node, end_node], relationships=[relationship])

    embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    neo4j_writer = Neo4jWriter(driver=driver, embedder=embedder)
    await neo4j_writer.run(graph=graph)

    query = """
    MATCH (a:Document {id: 1})-[r:NEXT_CHUNK]-(b:Document {id: 2})
    RETURN a, r, b
    """
    record = driver.execute_query(query).records[0]
    assert "a" and "b" and "r" in record.keys()

    node_a = record["a"]
    assert start_node.label == list(node_a.labels)[0]
    assert start_node.id == str(node_a.get("id"))
    if start_node.properties:
        for prop in start_node.properties:
            assert prop.key in node_a.keys()
            assert prop.value == node_a.get(prop.key)
    if start_node.embedding_properties:
        for embedding_prop in start_node.embedding_properties:
            assert embedding_prop.key in node_a.keys()
            assert len(node_a.get(embedding_prop.key)) == 384

    node_b = record["b"]
    assert end_node.label == list(node_b.labels)[0]
    assert end_node.id == str(node_b.get("id"))
    if end_node.properties:
        for prop in end_node.properties:
            assert prop.key in node_b.keys()
            assert prop.value == node_b.get(prop.key)
    if end_node.embedding_properties:
        for embedding_prop in end_node.embedding_properties:
            assert embedding_prop.key in node_b.keys()
            assert len(node_b.get(embedding_prop.key)) == 384

    rel = record["r"]
    assert rel.type == relationship.label
    assert relationship.start_node_id and relationship.end_node_id in [
        str(node.get("id")) for node in rel.nodes
    ]
