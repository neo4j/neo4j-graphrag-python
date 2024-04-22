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

from typing import Any, Literal, Optional
from pydantic import BaseModel, PositiveInt, model_validator, field_validator
from neo4j import Driver


class VectorSearchRecord(BaseModel):
    node: Any
    score: float


class EmbeddingVector(BaseModel):
    vector: list[float]


class IndexModel(BaseModel):
    driver: Any

    @field_validator("driver")
    def check_driver_is_valid(cls, v):
        if not isinstance(v, Driver):
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


class SimilaritySearchModel(BaseModel):
    index_name: str
    top_k: PositiveInt = 5
    query_vector: Optional[list[float]] = None
    query_text: Optional[str] = None

    @model_validator(mode="before")
    def check_query(cls, values):
        """
        Validates that one of either query_vector or query_text is provided exclusively.
        """
        query_vector, query_text = values.get("query_vector"), values.get("query_text")
        if not (bool(query_vector) ^ bool(query_text)):
            raise ValueError(
                "You must provide exactly one of query_vector or query_text."
            )
        return values


class VectorCypherSearchModel(SimilaritySearchModel):
    query_params: Optional[dict[str, Any]] = None
