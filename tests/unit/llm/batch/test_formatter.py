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
import copy
import json
import logging
from typing import Any, Literal

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from neo4j_graphrag.llm.batch.formatter import (
    VertexBatchRequestFormatter,
    VertexModelParams,
    _adapt_schema_for_vertex_ai,
    _resolve_json_schema_refs,
)


def test_adapt_schema_for_vertex_ai_converts_const_to_enum() -> None:
    schema = {"type": "object", "properties": {"kind": {"const": "movie"}}}

    adapted = _adapt_schema_for_vertex_ai(schema)

    assert adapted["properties"]["kind"] == {"enum": ["movie"]}


def test_adapt_schema_for_vertex_ai_converts_nullable_any_of() -> None:
    schema = {
        "properties": {
            "director": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "default": None,
            }
        }
    }

    adapted = _adapt_schema_for_vertex_ai(schema)

    assert adapted["properties"]["director"] == {
        "type": "string",
        "nullable": True,
    }


def test_adapt_schema_for_vertex_ai_leaves_non_nullable_any_of_untouched() -> None:
    # Only a two-branch anyOf with exactly one non-null branch is a nullable
    # field; anything else (e.g. a real union) has no Vertex AI equivalent
    # and is passed through as-is.
    schema = {
        "properties": {"value": {"anyOf": [{"type": "string"}, {"type": "integer"}]}}
    }

    adapted = _adapt_schema_for_vertex_ai(schema)

    assert adapted["properties"]["value"] == schema["properties"]["value"]


def test_adapt_schema_for_vertex_ai_fixes_nested_dicts_and_lists() -> None:
    schema = {
        "properties": {
            "items": {
                "type": "array",
                "items": {"const": "fixed"},
            },
            "variants": {
                "anyOf": [
                    {"type": "string"},
                    {"type": "null"},
                ],
                "default": None,
            },
        }
    }

    adapted = _adapt_schema_for_vertex_ai(schema)

    assert adapted["properties"]["items"]["items"] == {"enum": ["fixed"]}
    assert adapted["properties"]["variants"] == {"type": "string", "nullable": True}


def test_adapt_schema_for_vertex_ai_fixes_defs() -> None:
    schema = {
        "$defs": {
            "Genre": {"const": "drama"},
        },
        "properties": {"genre": {"$ref": "#/$defs/Genre"}},
    }

    adapted = _adapt_schema_for_vertex_ai(schema)

    assert adapted["$defs"]["Genre"] == {"enum": ["drama"]}


def test_adapt_schema_for_vertex_ai_does_not_mutate_input() -> None:
    schema: dict[str, Any] = {
        "$defs": {"Genre": {"const": "drama"}},
        "properties": {
            "kind": {"const": "movie"},
            "director": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "default": None,
            },
        },
    }
    original = copy.deepcopy(schema)

    _adapt_schema_for_vertex_ai(schema)

    assert schema == original


def test_adapt_schema_for_vertex_ai_returns_new_object() -> None:
    schema = {"properties": {"kind": {"const": "movie"}}}

    adapted = _adapt_schema_for_vertex_ai(schema)

    assert adapted is not schema
    assert adapted["properties"] is not schema["properties"]


# ---------------------------------------------------------------------------
# _resolve_json_schema_refs
# ---------------------------------------------------------------------------


def test_resolve_json_schema_refs_inlines_defs() -> None:
    class Genre(BaseModel):
        name: str

    class Movie(BaseModel):
        genre: Genre

    resolved = _resolve_json_schema_refs(Movie.model_json_schema())

    assert "$defs" not in resolved
    genre = resolved["properties"]["genre"]
    assert "$ref" not in genre
    assert genre["properties"]["name"] == {"title": "Name", "type": "string"}


def test_resolve_json_schema_refs_rejects_recursive_models() -> None:
    class Node(BaseModel):
        children: list["Node"] = []

    with pytest.raises(ValueError, match="Recursive"):
        _resolve_json_schema_refs(Node.model_json_schema())


def test_resolve_json_schema_refs_passes_through_unresolvable_refs() -> None:
    schema = {"properties": {"ext": {"$ref": "https://example.com/schema.json"}}}

    assert _resolve_json_schema_refs(schema) == schema


# ---------------------------------------------------------------------------
# BatchModelParams.from_llm_params / VertexModelParams.to_request_fields
# ---------------------------------------------------------------------------


def test_from_llm_params_accepts_camel_case_and_aliased_keys() -> None:
    params = VertexModelParams.from_llm_params(
        {"maxOutputTokens": 256, "topP": 0.9, "topK": 40}
    )

    assert params.max_tokens == 256
    assert params.top_p == 0.9
    assert params.top_k == 40


def test_from_llm_params_none_gives_defaults() -> None:
    params = VertexModelParams.from_llm_params(None)

    assert params.temperature is None
    assert params.stop_sequences == ()


def test_from_llm_params_sets_structured_output_naming() -> None:
    params = VertexModelParams.from_llm_params(
        None,
        structured_output_name="MovieInfo",
        structured_output_description="extracted movie",
    )

    assert params.structured_output_name == "MovieInfo"
    assert params.structured_output_description == "extracted movie"


def test_from_llm_params_warns_on_unrecognised_keys(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="neo4j_graphrag.llm.batch.formatter"):
        params = VertexModelParams.from_llm_params(
            {"temperature": 0.5, "not_a_param": 1}
        )

    assert params.temperature == 0.5
    assert "not_a_param" in caplog.text


def test_from_llm_params_drops_request_metadata_quietly(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # ``labels`` is request-level metadata with no batch field; dropping it
    # must not trip the unrecognised-key warning.
    with caplog.at_level(logging.WARNING, logger="neo4j_graphrag.llm.batch.formatter"):
        VertexModelParams.from_llm_params({"labels": {"team": "rag"}})

    assert caplog.text == ""


def test_from_llm_params_rejects_values_the_provider_cannot_use() -> None:
    with pytest.raises(ValidationError):
        VertexModelParams.from_llm_params({"max_tokens": 0})


def test_to_request_fields_omits_unset_params() -> None:
    assert VertexModelParams().to_request_fields() == {}


def test_to_request_fields_renders_provider_field_names() -> None:
    fields = VertexModelParams(
        temperature=0.0,
        max_tokens=10,
        top_p=0.5,
        top_k=5,
        stop_sequences=("END",),
        candidate_count=1,
        seed=7,
        response_mime_type="text/plain",
    ).to_request_fields()

    assert fields == {
        "temperature": 0.0,
        "maxOutputTokens": 10,
        "topP": 0.5,
        "topK": 5,
        "candidateCount": 1,
        "seed": 7,
        "responseMimeType": "text/plain",
        "stopSequences": ["END"],
    }


# ---------------------------------------------------------------------------
# VertexBatchRequestFormatter.format
# ---------------------------------------------------------------------------


def _generation_config(line: str) -> dict[str, Any]:
    return json.loads(line)["request"]["generationConfig"]  # type: ignore[no-any-return]


def test_format_renders_roles_and_echoes_key() -> None:
    formatter = VertexBatchRequestFormatter()

    result = formatter.format(
        "req-1",
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ],
    )

    assert result.key == "req-1"
    record = json.loads(result.line)
    assert record["key"] == "req-1"
    assert record["request"]["contents"] == [
        {"role": "user", "parts": [{"text": "hi"}]},
        {"role": "model", "parts": [{"text": "hello"}]},
    ]
    assert "system_instruction" not in record["request"]


def test_format_moves_system_messages_to_system_instruction() -> None:
    formatter = VertexBatchRequestFormatter()

    result = formatter.format(
        "req-2",
        [
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "hi"},
            {"role": "system", "content": "answer in json"},
        ],
    )

    request = json.loads(result.line)["request"]
    assert [content["role"] for content in request["contents"]] == ["user"]
    assert request["system_instruction"] == {
        "parts": [{"text": "be terse"}, {"text": "answer in json"}]
    }


def test_format_applies_model_params_to_generation_config() -> None:
    formatter = VertexBatchRequestFormatter(
        model_params=VertexModelParams(temperature=0.0, max_tokens=16)
    )

    config = _generation_config(
        formatter.format("k", [{"role": "user", "content": "x"}]).line
    )

    assert config["temperature"] == 0.0
    assert config["maxOutputTokens"] == 16
    assert "responseMimeType" not in config
    assert "responseSchema" not in config


class _Genre(BaseModel):
    name: str


class _Movie(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    genre: _Genre
    media_type: Literal["movie"]
    tagline: str | None = None


def test_format_defaults_mime_type_to_json_when_response_format_given() -> None:
    formatter = VertexBatchRequestFormatter(response_format=_Movie)

    config = _generation_config(
        formatter.format("k", [{"role": "user", "content": "x"}]).line
    )

    assert config["responseMimeType"] == "application/json"


def test_format_keeps_explicit_mime_type_when_response_format_given() -> None:
    formatter = VertexBatchRequestFormatter(
        model_params=VertexModelParams(response_mime_type="text/x.enum"),
        response_format=_Movie,
    )

    config = _generation_config(
        formatter.format("k", [{"role": "user", "content": "x"}]).line
    )

    assert config["responseMimeType"] == "text/x.enum"


def test_format_preserves_open_additional_properties_in_response_schema() -> None:
    formatter = VertexBatchRequestFormatter(
        response_format={
            "type": "object",
            "properties": {
                "tags": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                },
            },
            "additionalProperties": False,
        }
    )

    schema = _generation_config(
        formatter.format("k", [{"role": "user", "content": "x"}]).line
    )["responseSchema"]

    assert "additionalProperties" not in schema
    assert schema["properties"]["tags"] == {
        "type": "OBJECT",
        "additionalProperties": {"type": "STRING"},
    }


def test_format_adapts_pydantic_response_format_for_batch_proto() -> None:
    formatter = VertexBatchRequestFormatter(response_format=_Movie)

    schema = _generation_config(
        formatter.format("k", [{"role": "user", "content": "x"}]).line
    )["responseSchema"]

    assert schema["type"] == "OBJECT"
    assert "additionalProperties" not in schema
    assert "$defs" not in schema
    properties = schema["properties"]
    # A nested submodel's $ref is inlined.
    assert "$ref" not in properties["genre"]
    assert properties["genre"]["type"] == "OBJECT"
    assert properties["genre"]["properties"]["name"] == {
        "title": "Name",
        "type": "STRING",
    }
    # A single-value Literal's const becomes a single-value enum.
    assert properties["media_type"] == {
        "title": "Media Type",
        "type": "STRING",
        "enum": ["movie"],
    }
    # An optional field's anyOf-with-null becomes nullable, default dropped.
    assert properties["tagline"] == {
        "title": "Tagline",
        "type": "STRING",
        "nullable": True,
    }


def test_format_accepts_plain_dict_response_format() -> None:
    formatter = VertexBatchRequestFormatter(
        response_format={"type": "object", "properties": {"x": {"type": "string"}}}
    )

    schema = _generation_config(
        formatter.format("k", [{"role": "user", "content": "x"}]).line
    )["responseSchema"]

    assert schema["type"] == "OBJECT"
    assert schema["properties"]["x"]["type"] == "STRING"
