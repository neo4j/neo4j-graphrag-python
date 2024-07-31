import jinja2.exceptions
import pytest
from neo4j_genai.exceptions import PromptMissingInputError
from neo4j_genai.generation.prompts import PromptTemplate


def test_prompt_template_all_default() -> None:
    class MyTemplate(PromptTemplate):
        DEFAULT_TEMPLATE = "My question is {{ question }}"

    template = MyTemplate()
    assert template._string_template == MyTemplate.DEFAULT_TEMPLATE
    assert template.format(question="question") == "My question is question"


def test_prompt_template_overwrite_defaults() -> None:
    class MyTemplate(PromptTemplate):
        DEFAULT_TEMPLATE = "My question is {{question}}"

    template = MyTemplate(
        template="Please answer my question {{question}} as a {{speaker_type}}",
    )
    assert (
        template._string_template
        == "Please answer my question {{question}} as a {{speaker_type}}"
    )
    assert (
        template.format(question="question", speaker_type="child")
        == "Please answer my question question as a child"
    )


def test_prompt_template_format_missing_value() -> None:
    class MyTemplate(PromptTemplate):
        DEFAULT_TEMPLATE = """{{ value }}"""

    template = MyTemplate()
    with pytest.raises(jinja2.exceptions.UndefinedError) as excinfo:
        template.format(question="question")
    assert "'value' is undefined" in str(excinfo)


def test_prompt_template_format_extra_values() -> None:
    class MyTemplate(PromptTemplate):
        DEFAULT_TEMPLATE = "{{ value }}"

    template = MyTemplate()
    assert template.format(value="value") == "value"
