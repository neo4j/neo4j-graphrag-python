#  Copyright (c) "Neo4j"
#  Neo4j Sweden AB [https://neo4j.com]
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#      https://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import json
import os
import random
import string
from typing import Generator

import neo4j
import pytest
from neo4j.exceptions import Neo4jError

from neo4j_graphrag.retrievers import CypherRetriever
from neo4j_graphrag.types import RetrieverResultItem


# Fixture to create test data
@pytest.fixture
def sample_data(driver: neo4j.Driver) -> Generator[str, None, None]:
    # Generate a random prefix for category names to avoid conflicts between test runs
    prefix = ''.join(random.choices(string.ascii_lowercase, k=8))
    category_name = f"Category_{prefix}"
    
    # Create test data
    try:
        with driver.session() as session:
            session.run(
                """
                CREATE (c:Category {name: $category_name})
                CREATE (p1:Product {name: "Product1", price: 10.99, stock: 100, featured: true})
                CREATE (p2:Product {name: "Product2", price: 25.50, stock: 50, featured: false})
                CREATE (p3:Product {name: "Product3", price: 5.99, stock: 200, featured: true})
                CREATE (p1)-[:BELONGS_TO]->(c)
                CREATE (p2)-[:BELONGS_TO]->(c)
                CREATE (p3)-[:BELONGS_TO]->(c)
                """,
                category_name=category_name
            )
    except Neo4jError as e:
        pytest.fail(f"Failed to create test data: {e}")
    
    yield category_name
    
    # Clean up test data
    try:
        with driver.session() as session:
            session.run(
                """
                MATCH (p:Product)-[:BELONGS_TO]->(c:Category {name: $category_name})
                DETACH DELETE p, c
                """,
                category_name=category_name
            )
    except Neo4jError as e:
        pytest.fail(f"Failed to clean up test data: {e}")


def test_cypher_retriever_basic_query(driver: neo4j.Driver, sample_data: str) -> None:
    """Test basic query with CypherRetriever."""
    retriever = CypherRetriever(
        driver=driver,
        query="MATCH (p:Product) WHERE p.price > $min_price RETURN p ORDER BY p.price",
        parameters={
            "min_price": {
                "type": "number",
                "description": "Minimum product price"
            }
        }
    )
    
    # Execute the query
    result = retriever.search(parameters={"min_price": 10.0})
    
    # Verify the results
    assert len(result.items) == 2
    assert "Product1" in result.items[0].content or "Product2" in result.items[0].content
    assert "cypher" in result.metadata


def test_cypher_retriever_multiple_parameters(driver: neo4j.Driver, sample_data: str) -> None:
    """Test query with multiple parameters."""
    retriever = CypherRetriever(
        driver=driver,
        query="""
        MATCH (p:Product)
        WHERE p.price >= $min_price AND p.price <= $max_price
        AND p.stock > $min_stock
        RETURN p
        """,
        parameters={
            "min_price": {
                "type": "number",
                "description": "Minimum product price"
            },
            "max_price": {
                "type": "number",
                "description": "Maximum product price"
            },
            "min_stock": {
                "type": "integer",
                "description": "Minimum stock quantity"
            }
        }
    )
    
    # Execute the query with parameters
    result = retriever.search(parameters={
        "min_price": 5.0,
        "max_price": 15.0,
        "min_stock": 50
    })
    
    # Verify the results
    assert len(result.items) == 2
    assert any("Product1" in item.content for item in result.items)
    assert any("Product3" in item.content for item in result.items)


def test_cypher_retriever_optional_parameters(driver: neo4j.Driver, sample_data: str) -> None:
    """Test query with optional parameters."""
    retriever = CypherRetriever(
        driver=driver,
        query="""
        MATCH (p:Product)
        WHERE ($featured IS NULL OR p.featured = $featured)
        RETURN p
        """,
        parameters={
            "featured": {
                "type": "boolean",
                "description": "Filter for featured products",
                "required": False
            }
        }
    )
    
    # Execute the query with the optional parameter
    result_with_param = retriever.search(parameters={"featured": True})
    
    # Verify the results with parameter
    assert len(result_with_param.items) == 2
    assert all("featured: true" in item.content for item in result_with_param.items)
    
    # Execute the query without the optional parameter
    result_without_param = retriever.search(parameters={})
    
    # Verify the results without parameter (should return all products)
    assert len(result_without_param.items) == 3


def test_cypher_retriever_relationship_traversal(driver: neo4j.Driver, sample_data: str) -> None:
    """Test query with relationship traversal."""
    retriever = CypherRetriever(
        driver=driver,
        query="""
        MATCH (p:Product)-[:BELONGS_TO]->(c:Category {name: $category_name})
        RETURN p.name as product, p.price as price, c.name as category
        """,
        parameters={
            "category_name": {
                "type": "string",
                "description": "Category name"
            }
        }
    )
    
    # Execute the query
    result = retriever.search(parameters={"category_name": sample_data})
    
    # Verify the results
    assert len(result.items) == 3
    assert all(sample_data in item.content for item in result.items)


def test_cypher_retriever_custom_formatter(driver: neo4j.Driver, sample_data: str) -> None:
    """Test query with custom result formatter."""
    # Custom formatter that extracts product info in a structured format
    def product_formatter(record):
        product = record["p"]
        return RetrieverResultItem(
            content=f"{product['name']} - ${product['price']}",
            metadata={
                "price": product["price"],
                "stock": product["stock"],
                "featured": product["featured"]
            }
        )
    
    retriever = CypherRetriever(
        driver=driver,
        query="MATCH (p:Product) RETURN p",
        parameters={},
        result_formatter=product_formatter
    )
    
    # Execute the query
    result = retriever.search(parameters={})
    
    # Verify the results
    assert len(result.items) == 3
    
    # Check custom formatting
    for item in result.items:
        assert " - $" in item.content
        assert "price" in item.metadata
        assert "stock" in item.metadata
        assert "featured" in item.metadata