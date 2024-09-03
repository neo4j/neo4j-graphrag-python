import pytest
from neo4j_graphrag.exceptions import PromptMissingInputError
from neo4j_graphrag.generation.prompts import PromptTemplate


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
        expected_inputs=["query_text", "spaker_type"],
    )
    assert (
        template.template
        == "Please answer my question {query_text} as a {speaker_type}"
    )
    assert template.expected_inputs == ["query_text", "spaker_type"]
    assert (
        template.format(query_text="query_text", speaker_type="child")
        == "Please answer my question query_text as a child"
    )


def test_prompt_template_format_missing_value() -> None:
    class MyTemplate(PromptTemplate):
        EXPECTED_INPUTS = ["query_text", "other"]

    template = MyTemplate()
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
