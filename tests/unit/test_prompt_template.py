import pytest
from neo4j_genai.exceptions import PromptMissingInputError
from neo4j_genai.generation.prompts import PromptTemplate


def test_prompt_template_all_default() -> None:
    class MyTemplate(PromptTemplate):
        DEFAULT_TEMPLATE = "My question is {question}"
        EXPECTED_INPUTS = ["question"]

    template = MyTemplate()
    assert template.template == MyTemplate.DEFAULT_TEMPLATE
    assert template.expected_inputs == MyTemplate.EXPECTED_INPUTS
    assert template.format(question="question") == "My question is question"


def test_prompt_template_overwrite_defaults() -> None:
    class MyTemplate(PromptTemplate):
        DEFAULT_TEMPLATE = "My question is {question}"
        EXPECTED_INPUTS = ["question"]

    template = MyTemplate(
        template="Please answer my questeion {question} as a {speaker_type}",
        expected_inputs=["question", "spaker_type"],
    )
    assert (
        template.template == "Please answer my questeion {question} as a {speaker_type}"
    )
    assert template.expected_inputs == ["question", "spaker_type"]
    assert (
        template.format(question="question", speaker_type="child")
        == "Please answer my questeion question as a child"
    )


def test_prompt_template_format_missing_value() -> None:
    class MyTemplate(PromptTemplate):
        EXPECTED_INPUTS = ["question", "other"]

    template = MyTemplate()
    with pytest.raises(PromptMissingInputError) as excinfo:
        template.format(question="question")
    assert "Missing input 'other'" in str(excinfo)


def test_prompt_template_format_extra_values() -> None:
    class MyTemplate(PromptTemplate):
        DEFAULT_TEMPLATE = "My question is {question} {other}"
        EXPECTED_INPUTS = ["question"]

    template = MyTemplate()
    with pytest.raises(KeyError) as excinfo:
        template.format(question="question")
    assert "KeyError('other')" in str(excinfo)
