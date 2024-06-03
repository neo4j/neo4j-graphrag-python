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
import re
from unittest import mock
from unittest.mock import MagicMock

import pytest
from langchain_community.embeddings import HuggingFaceEmbeddings
from neo4j_genai.retrievers.external.pinecone import PineconeNeo4jRetriever
from neo4j_genai.types import RetrieverResult, RetrieverResultItem
from pinecone import Pinecone

from ..utils import EMBEDDING_BIOLOGY
from .populate_dbs import build_data_objects, populate_neo4j


@pytest.fixture(scope="module")
def sentence_transformer_embedder():
    embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    yield embedder


@pytest.fixture(scope="module")
def client():
    return MagicMock(spec=Pinecone)


@pytest.fixture(scope="module")
def populate_neo4j_db(driver):
    driver.execute_query("MATCH (n) DETACH DELETE n")
    neo4j_objects, _ = build_data_objects()
    populate_neo4j(driver, neo4j_objects)


@pytest.mark.usefixtures("populate_neo4j_db")
def test_pinecone_neo4j_vector_input(driver, client):
    retriever = PineconeNeo4jRetriever(
        driver=driver, client=client, index_name="jeopardy", id_property_neo4j="id"
    )
    with mock.patch.object(retriever, "index") as mock_index:
        top_k = 2
        mock_index.query.return_value = {
            "matches": [
                {
                    "id": "question_c458c6f64d8d47429636bc5a94c97f51",
                    "score": 0.232427984,
                    "values": [],
                },
                {
                    "id": "question_3d53154d16068c1e86e024923bc220d8",
                    "score": 0.184265107,
                    "values": [],
                },
            ],
            "namespace": "",
            "usage": {"read_units": 5},
        }

        results = retriever._get_search_results(
            query_vector=EMBEDDING_BIOLOGY, top_k=top_k
        )

        assert isinstance(results, RetrieverResult)
        assert len(results.items) == top_k
        for result in results:
            assert isinstance(result, RetrieverResultItem)
        pattern = (
            r"<Record node=<Node element_id='.+' "
            "labels=frozenset\({'Question'}\) properties={'question': 'In 1953 Watson \& "
            "Crick built a model of the molecular structure of this, the gene-carrying "
            "substance', 'id': 'question_c458c6f64d8d47429636bc5a94c97f51'}> "
            r"score=0.232427984[0-9]+>"
        )
        assert re.match(pattern, results.items[0].content)


@pytest.mark.usefixtures("populate_neo4j_db")
def test_pinecone_neo4j_text_input(driver, client, sentence_transformer_embedder):
    retriever = PineconeNeo4jRetriever(
        driver=driver,
        client=client,
        index_name="jeopardy",
        id_property_neo4j="id",
        embedder=sentence_transformer_embedder,
    )
    with mock.patch.object(retriever, "index") as mock_index:
        top_k = 2
        mock_index.query.return_value = {
            "matches": [
                {
                    "id": "question_c458c6f64d8d47429636bc5a94c97f51",
                    "score": 0.232427984,
                    "values": [],
                },
                {
                    "id": "question_3d53154d16068c1e86e024923bc220d8",
                    "score": 0.184265107,
                    "values": [],
                },
            ],
            "namespace": "",
            "usage": {"read_units": 5},
        }

        results = retriever.search(query_text="biology", top_k=top_k)

        assert isinstance(results, RetrieverResult)
        assert len(results.items) == top_k
        for result in results:
            assert isinstance(result, RetrieverResultItem)
        pattern = (
            r"<Record node=<Node element_id='.+' "
            "labels=frozenset\({'Question'}\) properties={'question': 'In 1953 Watson \& "
            "Crick built a model of the molecular structure of this, the gene-carrying "
            "substance', 'id': 'question_c458c6f64d8d47429636bc5a94c97f51'}> "
            r"score=0.232427984[0-9]+>"
        )
        assert re.match(pattern, results.items[0].content)
