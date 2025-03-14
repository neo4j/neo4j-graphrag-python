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

"""Example of using CypherRetriever for parametrized Cypher queries.

This example demonstrates how to use CypherRetriever to define a retriever with
a templated Cypher query that accepts parameters at runtime.
"""

import neo4j
from neo4j_graphrag.retrievers import CypherRetriever
from neo4j_graphrag.types import RetrieverResultItem

# Connect to Neo4j
# Replace with your own connection details
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"  # Change this in production

driver = neo4j.GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# Simple example: Find a movie by title
def find_movie_by_title():
    retriever = CypherRetriever(
        driver=driver,
        query="MATCH (m:Movie {title: $movie_title}) RETURN m",
        parameters={
            "movie_title": {
                "type": "string",
                "description": "Title of a movie"
            }
        }
    )
    
    # Use the retriever to search for a movie
    result = retriever.search(parameters={"movie_title": "The Matrix"})
    
    print("=== Find Movie by Title ===")
    for item in result.items:
        print(f"Movie: {item.content}")
    print()


# Advanced example: Find movies with multiple criteria
def find_movies_by_criteria():
    # Custom formatter to extract specific information
    def movie_formatter(record):
        movie = record["m"]
        return RetrieverResultItem(
            content=f"{movie['title']} ({movie['released']})",
            metadata={
                "rating": movie.get("rating"),
                "tagline": movie.get("tagline"),
            }
        )
    
    # Create a more complex retriever with multiple parameters
    retriever = CypherRetriever(
        driver=driver,
        query="""
        MATCH (m:Movie)
        WHERE ($title IS NULL OR m.title CONTAINS $title)
        AND ($min_year IS NULL OR m.released >= $min_year)
        AND ($max_year IS NULL OR m.released <= $max_year)
        AND ($min_rating IS NULL OR m.rating >= $min_rating)
        RETURN m
        ORDER BY m.rating DESC
        LIMIT $limit
        """,
        parameters={
            "title": {
                "type": "string",
                "description": "Partial movie title to search for",
                "required": False
            },
            "min_year": {
                "type": "integer",
                "description": "Minimum release year",
                "required": False
            },
            "max_year": {
                "type": "integer",
                "description": "Maximum release year",
                "required": False
            },
            "min_rating": {
                "type": "number",
                "description": "Minimum movie rating",
                "required": False
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results to return",
                "required": True
            }
        },
        result_formatter=movie_formatter
    )
    
    # Search with optional parameters
    result = retriever.search(
        parameters={
            "title": "Matrix",
            "min_year": 1990,
            "min_rating": 7.5,
            "limit": 5
        }
    )
    
    print("=== Find Movies by Criteria ===")
    for item in result.items:
        print(f"Movie: {item.content}")
        if item.metadata:
            if "rating" in item.metadata:
                print(f"  Rating: {item.metadata['rating']}")
            if "tagline" in item.metadata:
                print(f"  Tagline: {item.metadata['tagline']}")
    print()


# Example with relationship traversal
def find_actors_in_movie():
    retriever = CypherRetriever(
        driver=driver,
        query="""
        MATCH (m:Movie {title: $movie_title})<-[r:ACTED_IN]-(a:Person)
        RETURN a.name as actor, r.roles as roles
        ORDER BY a.name
        """,
        parameters={
            "movie_title": {
                "type": "string",
                "description": "Title of a movie"
            }
        }
    )
    
    result = retriever.search(parameters={"movie_title": "The Matrix"})
    
    print("=== Find Actors in Movie ===")
    for item in result.items:
        record = eval(item.content)  # Simple way to parse the string representation
        actor = record.get("actor", "Unknown")
        roles = record.get("roles", [])
        roles_str = ", ".join(roles) if roles else "Unknown role"
        print(f"Actor: {actor} as {roles_str}")
    print()


if __name__ == "__main__":
    try:
        # Setup: Make sure we have some movie data
        with driver.session() as session:
            # Check if data exists
            result = session.run("MATCH (m:Movie) RETURN count(m) as count")
            count = result.single()["count"]
            
            if count == 0:
                print("No movie data found. Creating sample data...")
                # Create sample data if none exists
                session.run("""
                CREATE (TheMatrix:Movie {title:'The Matrix', released:1999, tagline:'Welcome to the Real World', rating: 8.7})
                CREATE (Keanu:Person {name:'Keanu Reeves', born:1964})
                CREATE (Carrie:Person {name:'Carrie-Anne Moss', born:1967})
                CREATE (Laurence:Person {name:'Laurence Fishburne', born:1961})
                CREATE (Hugo:Person {name:'Hugo Weaving', born:1960})
                CREATE (Keanu)-[:ACTED_IN {roles:['Neo']}]->(TheMatrix)
                CREATE (Carrie)-[:ACTED_IN {roles:['Trinity']}]->(TheMatrix)
                CREATE (Laurence)-[:ACTED_IN {roles:['Morpheus']}]->(TheMatrix)
                CREATE (Hugo)-[:ACTED_IN {roles:['Agent Smith']}]->(TheMatrix)
                CREATE (TheMatrixReloaded:Movie {title:'The Matrix Reloaded', released:2003, tagline:'Free your mind', rating: 7.2})
                CREATE (Keanu)-[:ACTED_IN {roles:['Neo']}]->(TheMatrixReloaded)
                CREATE (Carrie)-[:ACTED_IN {roles:['Trinity']}]->(TheMatrixReloaded)
                CREATE (Laurence)-[:ACTED_IN {roles:['Morpheus']}]->(TheMatrixReloaded)
                CREATE (Hugo)-[:ACTED_IN {roles:['Agent Smith']}]->(TheMatrixReloaded)
                CREATE (TheMatrixRevolutions:Movie {title:'The Matrix Revolutions', released:2003, tagline:'Everything that has a beginning has an end', rating: 6.8})
                CREATE (Keanu)-[:ACTED_IN {roles:['Neo']}]->(TheMatrixRevolutions)
                CREATE (Carrie)-[:ACTED_IN {roles:['Trinity']}]->(TheMatrixRevolutions)
                CREATE (Laurence)-[:ACTED_IN {roles:['Morpheus']}]->(TheMatrixRevolutions)
                CREATE (Hugo)-[:ACTED_IN {roles:['Agent Smith']}]->(TheMatrixRevolutions)
                """)
                print("Sample data created.")
            else:
                print(f"Found {count} movies in the database.")
        
        # Run the examples
        find_movie_by_title()
        find_movies_by_criteria()
        find_actors_in_movie()
        
    finally:
        # Close the driver
        driver.close()