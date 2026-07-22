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
import pytest
from neo4j_graphrag.embeddings.base import Embedder


class MinimalEmbedder(Embedder):
    """Concrete subclass for testing the base class."""

    def embed_query(self, text: str) -> list[float]:
        return []


def test_embedder_stores_model_and_dimensions() -> None:
    embedder = MinimalEmbedder(model="my-model", dimensions=512)
    assert embedder.model == "my-model"
    assert embedder.dimensions == 512


def test_embedder_dimensions_defaults_to_none() -> None:
    embedder = MinimalEmbedder(model="my-model")
    assert embedder.dimensions is None


def test_embedder_empty_model_raises_deprecation_warning() -> None:
    with pytest.warns(DeprecationWarning, match="will be required in 2.0"):
        MinimalEmbedder(model="")


def test_embedder_omitted_model_raises_deprecation_warning() -> None:
    with pytest.warns(DeprecationWarning, match="will be required in 2.0"):
        MinimalEmbedder()


def test_embedder_non_string_model_raises_type_error() -> None:
    """Catches the silent failure of passing rate_limit_handler positionally."""
    with pytest.raises(TypeError, match="'model' must be a str"):
        MinimalEmbedder(object())  # type: ignore[arg-type]
