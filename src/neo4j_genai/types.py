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
from enum import Enum
from typing import Any, Literal, Optional

import neo4j
from pydantic import (
    BaseModel,
    ConfigDict,
    PositiveInt,
    field_validator,
    model_validator,
)
import neo4j
from neo4j_genai.retrievers.utils import validate_search_query_input


class VectorSearchRecord(BaseModel):
    """
    Represents a record returned from a vector search.

    Attributes:
        node (Any): The node data retrieved by the search.
        score (float): The similarity score of the retrieved embedding.
    """

    node: Any
    score: float


class Neo4jSchemaModel(BaseModel):
    neo4j_schema: str


class IndexModel(BaseModel):
    driver: Any

    @field_validator("driver")
    def check_driver_is_valid(cls, v):
        if not isinstance(v, neo4j.Driver):
            raise ValueError("driver must be an instance of neo4j.Driver")
        return v


class VectorIndexModel(IndexModel):
    name: str
    label: str
    property: str
    dimensions: PositiveInt
    similarity_fn: Literal["euclidean", "cosine"]


class FulltextIndexModel(IndexModel):
    name: str
    label: str
    node_properties: list[str]

    @field_validator("node_properties")
    def check_node_properties_not_empty(cls, v):
        if len(v) == 0:
            raise ValueError("node_properties cannot be an empty list")
        return v


class VectorSearchModel(BaseModel):
    vector_index_name: str
    top_k: PositiveInt = 5
    query_vector: Optional[list[float]] = None
    query_text: Optional[str] = None

    @model_validator(mode="before")
    def check_query(cls, values):
        """
        Validates that one of either query_vector or query_text is provided exclusively.
        """
        query_vector, query_text = values.get("query_vector"), values.get("query_text")
        validate_search_query_input(query_text, query_vector)
        return values


class VectorCypherSearchModel(VectorSearchModel):
    query_params: Optional[dict[str, Any]] = None


class HybridSearchModel(BaseModel):
    vector_index_name: str
    fulltext_index_name: str
    query_text: str
    top_k: PositiveInt = 5
    query_vector: Optional[list[float]] = None


class HybridCypherSearchModel(HybridSearchModel):
    query_params: Optional[dict[str, Any]] = None


class Text2CypherSearchModel(BaseModel):
    query_text: str
    examples: Optional[list[str]] = None


class SearchType(str, Enum):
    """Enumerator of the search strategies."""

    VECTOR = "vector"
    HYBRID = "hybrid"


class EmbedderModel(BaseModel):
    embedder: Optional[Any]
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("embedder")
    def check_embedder(cls, value):
        if not hasattr(value, "embed_query") or not callable(
            getattr(value, "embed_query", None)
        ):
            raise ValueError(
                "Provided embedder object must have an 'embed_query' callable method."
            )
        return value


class LLMModel(BaseModel):
    llm: Any
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("llm")
    def check_llm(cls, value):
        if not hasattr(value, "invoke") or not callable(getattr(value, "invoke", None)):
            raise ValueError(
                "Provided llm object must have an 'invoke' callable method."
            )
        return value


class Neo4jDriverModel(BaseModel):
    driver: neo4j.Driver
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("driver")
    def check_driver(cls, value):
        if not isinstance(value, neo4j.Driver):
            raise ValueError("Provided driver needs to be of type neo4j.Driver")
        return value


class VectorRetrieverModel(BaseModel):
    driver_model: Neo4jDriverModel
    index_name: str
    embedder_model: Optional[EmbedderModel] = None
    return_properties: Optional[list[str]] = None


class VectorCypherRetrieverModel(BaseModel):
    driver_model: Neo4jDriverModel
    index_name: str
    retrieval_query: str
    embedder_model: Optional[EmbedderModel] = None


class HybridRetrieverModel(BaseModel):
    driver_model: Neo4jDriverModel
    vector_index_name: str
    fulltext_index_name: str
    embedder_model: Optional[EmbedderModel] = None
    return_properties: Optional[list[str]] = None


class HybridCypherRetrieverModel(BaseModel):
    driver_model: Neo4jDriverModel
    vector_index_name: str
    fulltext_index_name: str
    retrieval_query: str
    embedder_model: Optional[EmbedderModel] = None


class Text2CypherRetrieverModel(BaseModel):
    driver_model: Neo4jDriverModel
    llm_model: LLMModel
    neo4j_schema_model: Optional[Neo4jSchemaModel] = None
