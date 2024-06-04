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


def populate_neo4j(neo4j_driver, neo4j_objs):
    question_nodes = list(
        filter(lambda x: x["label"] == "Question", neo4j_objs["nodes"])
    )
    answer_nodes = list(filter(lambda x: x["label"] == "Answer", neo4j_objs["nodes"]))
    category_nodes = list(
        filter(lambda x: x["label"] == "Category", neo4j_objs["nodes"])
    )
    belongs_to_relationships = list(
        filter(lambda x: x["type"] == "BELONGS_TO", neo4j_objs["relationships"])
    )
    has_answer_relationships = list(
        filter(lambda x: x["type"] == "HAS_ANSWER", neo4j_objs["relationships"])
    )
    question_nodes_cypher = "UNWIND $nodes as node MERGE (n:Question {id: node.properties.id}) ON CREATE SET n = node.properties"
    answer_nodes_cypher = "UNWIND $nodes as node MERGE (n:Answer {id: node.properties.id}) ON CREATE SET n = node.properties"
    category_nodes_cypher = (
        "UNWIND $nodes as node MERGE (n:Category {id: node.id}) ON CREATE SET n = node"
    )
    belongs_to_relationships_cypher = "UNWIND $relationships as rel MATCH (q:Question {id: rel.start_node_id}), (c:Category {id: rel.end_node_id}) MERGE (q)-[r:BELONGS_TO]->(c)"
    has_answer_relationships_cypher = "UNWIND $relationships as rel MATCH (q:Question {id: rel.start_node_id}), (a:Answer {id: rel.end_node_id}) MERGE (q)-[r:HAS_ANSWER]->(a)"
    neo4j_driver.execute_query(question_nodes_cypher, {"nodes": question_nodes})
    neo4j_driver.execute_query(answer_nodes_cypher, {"nodes": answer_nodes})
    neo4j_driver.execute_query(category_nodes_cypher, {"nodes": category_nodes})
    neo4j_driver.execute_query(
        belongs_to_relationships_cypher, {"relationships": belongs_to_relationships}
    )
    res = neo4j_driver.execute_query(
        has_answer_relationships_cypher, {"relationships": has_answer_relationships}
    )
    return res
