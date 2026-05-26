import json

import pytest
from neo4j_graphrag.exceptions import (
    PromptMissingInputError,
    PromptMissingPlaceholderError,
)
from neo4j_graphrag.generation.prompts import ERExtractionTemplate, PromptTemplate


def test_prompt_template_all_default() -> None:
    class MyTemplate(PromptTemplate):
        DEFAULT_TEMPLATE = "My question is {query_text}"
        EXPECTED_INPUTS = ["query_text"]

    template = MyTemplate()
    assert template.template == MyTemplate.DEFAULT_TEMPLATE
    assert template.expected_inputs == MyTemplate.EXPECTED_INPUTS
    assert template.format(query_text="query_text") == "My question is query_text"


def test_prompt_template_overwrite_defaults() -> None:
    class MyTemplate(PromptTemplate):
        DEFAULT_TEMPLATE = "My question is {query_text}"
        EXPECTED_INPUTS = ["query_text"]

    template = MyTemplate(
        template="Please answer my question {query_text} as a {speaker_type}",
        expected_inputs=["query_text", "speaker_type"],
    )
    assert (
        template.template
        == "Please answer my question {query_text} as a {speaker_type}"
    )
    assert template.expected_inputs == ["query_text", "speaker_type"]
    assert (
        template.format(query_text="query_text", speaker_type="child")
        == "Please answer my question query_text as a child"
    )


def test_prompt_template_format_missing_value() -> None:
    class MyTemplate(PromptTemplate):
        EXPECTED_INPUTS = ["query_text", "other"]

    template = MyTemplate(template="{query_text} {other}")
    with pytest.raises(PromptMissingInputError) as excinfo:
        template.format(query_text="query_text")
    assert "Missing input 'other'" in str(excinfo)


def test_prompt_template_format_extra_values() -> None:
    class MyTemplate(PromptTemplate):
        DEFAULT_TEMPLATE = "My question is {query_text} {other}"
        EXPECTED_INPUTS = ["query_text"]

    template = MyTemplate()
    with pytest.raises(KeyError) as excinfo:
        template.format(query_text="query_text")
    assert "KeyError('other')" in str(excinfo)


def test_prompt_template_missing_placeholders() -> None:
    class MyTemplate(PromptTemplate):
        DEFAULT_TEMPLATE = "My question is {query_text} {other}"
        EXPECTED_INPUTS = ["query_text", "banana"]

    with pytest.raises(PromptMissingPlaceholderError) as e:
        MyTemplate()

    assert "`template` is missing placeholder banana" in str(e)


def test_prompt_template_format_given_unused_kwargs() -> None:
    class MyTemplate(PromptTemplate):
        DEFAULT_TEMPLATE = "My question is {query_text}"
        EXPECTED_INPUTS = ["query_text"]

    template = MyTemplate()

    assert (
        template.format(query_text="what do?", banana="b") == "My question is what do?"
    )


def test_er_extraction_template_no_existing_entities() -> None:
    template = ERExtractionTemplate()
    prompt = template.format(schema={}, examples="", text="some text")
    assert "Existing graph entities" not in prompt
    assert "None" not in prompt


def test_er_extraction_template_with_existing_nodes() -> None:
    template = ERExtractionTemplate()
    existing_nodes = [{"id": "0", "label": "Person", "properties": {"name": "Alice"}}]
    prompt = template.format(
        schema={},
        examples="",
        text="some text",
        existing_nodes=existing_nodes,
    )
    assert "Existing graph entities" in prompt
    assert json.dumps({"nodes": existing_nodes, "relationships": []}) in prompt


def test_er_extraction_template_with_existing_nodes_and_rels() -> None:
    template = ERExtractionTemplate()
    existing_nodes = [{"id": "0", "label": "Person", "properties": {"name": "Alice"}}]
    existing_rels = [
        {"type": "KNOWS", "start_node_id": "0", "end_node_id": "1", "properties": {}}
    ]
    prompt = template.format(
        schema={},
        examples="",
        text="some text",
        existing_nodes=existing_nodes,
        existing_rels=existing_rels,
    )
    assert "Existing graph entities" in prompt
    assert (
        json.dumps({"nodes": existing_nodes, "relationships": existing_rels}) in prompt
    )


def test_er_extraction_template_with_empty_existing_lists() -> None:
    template = ERExtractionTemplate()
    prompt = template.format(
        schema={},
        examples="",
        text="some text",
        existing_nodes=[],
        existing_rels=[],
    )
    assert "Existing graph entities" not in prompt
