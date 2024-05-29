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
from unittest.mock import patch

import pytest
from neo4j.exceptions import CypherSyntaxError, Neo4jError
from neo4j_genai import Text2CypherRetriever
from neo4j_genai.prompts import TEXT2CYPHER_PROMPT


def test_t2c_retriever_initialization(driver, llm):
    with patch("neo4j_genai.retrievers.base.Retriever._verify_version") as mock_verify:
        Text2CypherRetriever(driver, llm, neo4j_schema="dummy-text")
        mock_verify.assert_called_once()


@patch("neo4j_genai.retrievers.base.Retriever._verify_version")
@patch("neo4j_genai.retrievers.text2cypher.get_schema")
def test_t2c_retriever_schema_retrieval(
    _verify_version_mock, get_schema_mock, driver, llm
):
    Text2CypherRetriever(driver, llm)
    get_schema_mock.assert_called_once()


@patch("neo4j_genai.retrievers.base.Retriever._verify_version")
@patch("neo4j_genai.retrievers.text2cypher.get_schema")
def test_t2c_retriever_schema_retrieval_failure(
    _verify_version_mock, get_schema_mock, driver, llm
):
    get_schema_mock.side_effect = Neo4jError
    with pytest.raises(Neo4jError):
        Text2CypherRetriever(driver, llm)


@patch("neo4j_genai.Text2CypherRetriever._verify_version")
def test_t2c_retriever_invalid_neo4j_schema(_verify_version_mock, driver, llm):
    with pytest.raises(ValueError) as exc_info:
        Text2CypherRetriever(driver=driver, llm=llm, neo4j_schema=42)

    assert "neo4j_schema" in str(exc_info.value)
    assert "Input should be a valid string" in str(exc_info.value)


@patch("neo4j_genai.Text2CypherRetriever._verify_version")
def test_t2c_retriever_invalid_search_query(_verify_version_mock, driver, llm):
    with pytest.raises(ValueError) as exc_info:
        retriever = Text2CypherRetriever(
            driver=driver, llm=llm, neo4j_schema="dummy-text"
        )
        retriever.search(query_text=42)

    assert "query_text" in str(exc_info.value)
    assert "Input should be a valid string" in str(exc_info.value)


@patch("neo4j_genai.Text2CypherRetriever._verify_version")
def test_t2c_retriever_invalid_search_examples(_verify_version_mock, driver, llm):
    with pytest.raises(ValueError) as exc_info:
        retriever = Text2CypherRetriever(
            driver=driver, llm=llm, neo4j_schema="dummy-text"
        )
        retriever.search(query_text="dummy-text", examples=42)

    assert "examples" in str(exc_info.value)
    assert "Input should be a valid list" in str(exc_info.value)


@patch("neo4j_genai.Text2CypherRetriever._verify_version")
def test_t2c_retriever_happy_path(_verify_version_mock, driver, llm, neo4j_record):
    t2c_query = "MATCH (n) RETURN n;"
    query_text = "may thy knife chip and shatter"
    neo4j_schema = "dummy-schema"
    examples = ["example-1", "example-2"]
    retriever = Text2CypherRetriever(driver=driver, llm=llm, neo4j_schema=neo4j_schema)
    retriever.llm.invoke.return_value = t2c_query
    retriever.driver.execute_query.return_value = (
        [neo4j_record],
        None,
        None,
    )
    prompt = TEXT2CYPHER_PROMPT.format(
        schema=neo4j_schema,
        examples="\n".join(examples),
        input=query_text,
    )
    retriever.search(query_text=query_text, examples=examples)
    retriever.llm.invoke.assert_called_once_with(prompt)
    retriever.driver.execute_query.assert_called_once_with(query_=t2c_query)


@patch("neo4j_genai.Text2CypherRetriever._verify_version")
def test_t2c_retriever_cypher_error(_verify_version_mock, driver, llm):
    t2c_query = "this is not a cypher query"
    neo4j_schema = "dummy-schema"
    examples = ["example-1", "example-2"]
    retriever = Text2CypherRetriever(driver=driver, llm=llm, neo4j_schema=neo4j_schema)
    retriever.llm.invoke.return_value = t2c_query
    query_text = "may thy knife chip and shatter"
    driver.execute_query.side_effect = CypherSyntaxError
    with pytest.raises(RuntimeError) as e:
        retriever.search(query_text=query_text, examples=examples)
    assert "Cypher query generation failed" in str(e)
