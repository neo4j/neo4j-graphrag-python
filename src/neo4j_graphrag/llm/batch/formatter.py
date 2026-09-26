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
"""Builds the request lines a :class:`~neo4j_graphrag.llm.batch.client.BaseBatchClient`
submits, from ``LLMMessage`` lists.

:class:`~neo4j_graphrag.llm.batch.client.BaseBatchClient` only submits an
already-built ``requests.jsonl`` and polls the resulting job; it has no
opinion on how a request line is built.  :class:`BatchRequestFormatter` is the
counterpart that does — kept separate so request formatting (which depends on
the model, response schema, and generation params) doesn't leak into the
client (which depends only on transport/auth), and so a request built once
here can be handed to whichever client subclass ends up submitting it.

Each provider's batch prediction API has its own request envelope and its own
generation-parameter names, so this module pairs one
:class:`BatchRequestFormatter` subclass with one :class:`BatchModelParams`
subclass per provider, mirroring the one
:class:`~neo4j_graphrag.llm.batch.client.BaseBatchClient` subclass per
transport. :class:`VertexBatchRequestFormatter`/:class:`VertexModelParams` are
the Vertex AI Gemini pair; a further provider (e.g. Bedrock, whose structured
output is a forced tool call rather than a ``responseSchema``) is a further
pair here.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from re import sub as _re_sub
from typing import Any, ClassVar, cast

from typing_extensions import Self

from pydantic import BaseModel, ConfigDict, Field

from neo4j_graphrag.types import LLMMessage

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BatchRequestLine:
    """One formatted request, ready to write as a line of ``requests.jsonl``.

    Attributes:
        key: The correlation id passed in to :meth:`BatchRequestFormatter.format`,
            echoed back unchanged in the batch prediction response so a
            :class:`~neo4j_graphrag.llm.batch.reader.BaseBatchResponseReader`
            can join a result back to whatever produced *key*.
        line: The request, already JSON-encoded (``ensure_ascii=False``) and
            ready to write verbatim as one line of ``requests.jsonl``.
    """

    key: str
    line: str


# ---------------------------------------------------------------------------
# Model parameters
# ---------------------------------------------------------------------------


def _to_snake_case(key: str) -> str:
    """Rewrite a camelCase (or already snake_case) key to snake_case."""
    return _re_sub(r"(?<!^)(?=[A-Z])", "_", key).lower()


class BatchModelParams(BaseModel, ABC):
    """Normalised generation parameters for one provider's batch requests.

    Fields use neo4j_graphrag's own vocabulary (matching what an LLM's
    ``model_params`` mapping ordinarily holds) so a batch job's params can be
    built directly from an existing interactive LLM's ``model_params`` via
    :meth:`from_llm_params`, rather than requiring a caller to hand-translate
    into request-body field names. Subclasses render the validated params
    onto one provider's own field names via :meth:`to_request_fields`.

    Args:
        temperature: The temperature to use for the generation.
        max_tokens: The maximum number of tokens to generate.
        top_p: For top-p sampling. The cumulative probability mass of the top p tokens to sample from. (0.0 to 1.0)
        top_k: For top-k sampling. The number of tokens to sample from the top k tokens.
        stop_sequences: The stop sequences to use for the generation.
        structured_output_name: The name of the structured output to use for the generation.
        structured_output_description: The description of the structured output to use for the generation.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    temperature: float | None = Field(default=None, ge=0.0)
    max_tokens: int | None = Field(default=None, gt=0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    top_k: int | None = Field(default=None, gt=0)
    stop_sequences: tuple[str, ...] = ()
    structured_output_name: str | None = None
    structured_output_description: str | None = None

    # Loose input keys that name a field by another spelling, beyond the
    # camelCase-to-snake_case rewrite every key gets first.
    _KEY_ALIASES: ClassVar[Mapping[str, str]] = {"max_output_tokens": "max_tokens"}

    # Not generation parameters at all: ``labels`` is request-level metadata an
    # LLM factory can inject, which batch input has no field for. Dropped
    # quietly, so the warning below stays meaningful for genuinely unexpected
    # keys.
    _NON_GENERATION_KEYS: ClassVar[frozenset[str]] = frozenset({"labels"})

    @classmethod
    def from_llm_params(
        cls,
        model_params: Mapping[str, Any] | None,
        *,
        structured_output_name: str | None = None,
        structured_output_description: str | None = None,
    ) -> Self:
        """Normalise a loose ``model_params`` mapping onto this variant's fields.

        Args:
            model_params: An LLM's ``model_params``, in any supported spelling.
            structured_output_name: Name for the constrained-output mechanism,
                for the providers that give it one (see
                :attr:`structured_output_name`).
            structured_output_description: Description for the same.

        Returns:
            A validated instance of this variant.

        Raises:
            pydantic.ValidationError: If a recognised key holds a value this
                provider cannot use (a non-positive ``max_tokens``, say).
                Failing here beats submitting a whole job built on it.
        """
        placed: dict[str, Any] = {}
        unplaced: list[str] = []
        for key, value in (model_params or {}).items():
            name = _to_snake_case(key)
            name = cls._KEY_ALIASES.get(name, name)
            if name in cls.model_fields:
                placed[name] = value
            elif name not in cls._NON_GENERATION_KEYS:
                unplaced.append(key)
        if unplaced:
            logger.warning(
                "Ignoring model params %s: not generation parameters %s accepts. "
                "Accepted: %s",
                sorted(unplaced),
                cls.__name__,
                sorted(cls.model_fields),
            )
        if structured_output_name is not None:
            placed["structured_output_name"] = structured_output_name
        if structured_output_description is not None:
            placed["structured_output_description"] = structured_output_description
        return cls.model_validate(placed)

    @abstractmethod
    def to_request_fields(self) -> dict[str, Any]:
        """Render the configured params as this provider's own request keys.

        Unset params are omitted rather than sent as ``null``. Structured-output
        naming is not included: it is not a generation parameter, and only a
        provider that expresses constrained output as a forced tool call has
        anywhere to put it.
        """

    @staticmethod
    def _drop_unset(fields: Mapping[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in fields.items() if value is not None}


class VertexModelParams(BatchModelParams):
    """Gemini generation parameters, rendered as a ``generationConfig`` body.

    ``response_mime_type`` defaults to JSON: the batch path is only used for
    structured extraction, and Gemini otherwise returns prose.
    """

    response_mime_type: str = "application/json"
    candidate_count: int | None = Field(default=None, gt=0)
    seed: int | None = None

    def to_request_fields(self) -> dict[str, Any]:
        return self._drop_unset(
            {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_tokens,
                "topP": self.top_p,
                "topK": self.top_k,
                "candidateCount": self.candidate_count,
                "seed": self.seed,
                "responseMimeType": self.response_mime_type,
                "stopSequences": list(self.stop_sequences) or None,
            }
        )


# ---------------------------------------------------------------------------
# Response schema adaptation (Vertex AI)
# ---------------------------------------------------------------------------


def _resolve_json_schema_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Resolve ``$ref`` references in a JSON schema by inlining definitions.

    Vertex AI's batch-prediction ``responseSchema`` has no notion of ``$ref``/
    ``$defs``, so a schema built from a Pydantic model with a nested submodel
    (which ``model_json_schema()`` renders as a ``$defs``/``$ref`` pair) must
    have every reference inlined before it is sent.

    Args:
        schema: A JSON schema dict, e.g. from ``BaseModel.model_json_schema()``.

    Returns:
        A JSON schema with every ``$ref`` resolved/inlined and ``$defs`` dropped.
    """
    defs: dict[str, Any] = schema.get("$defs", {})
    return _resolve_refs(schema, defs, ())


def _resolve_refs(
    schema: Any, defs: dict[str, Any], expanding: tuple[str, ...]
) -> dict[str, Any]:
    """Recursively resolve ``$ref`` references in a JSON schema.

    ``expanding`` holds the definitions currently being inlined, so a
    self-referencing model raises instead of recursing forever.
    """
    if not isinstance(schema, dict):
        return cast(dict[str, Any], schema)

    schema_dict = cast(dict[str, Any], schema)

    if "$ref" in schema_dict:
        ref_path: str = schema_dict["$ref"]
        if ref_path.startswith("#/$defs/"):
            def_name = ref_path.split("/")[-1]
            if def_name in defs:
                if def_name in expanding:
                    raise ValueError(
                        f"Recursive response_format is not supported: "
                        f"'{def_name}' refers to itself."
                    )
                return _resolve_refs(dict(defs[def_name]), defs, (*expanding, def_name))
        return dict(schema_dict)

    result: dict[str, Any] = {}
    for key, value in schema_dict.items():
        if key == "$defs":
            continue
        elif isinstance(value, dict):
            result[key] = _resolve_refs(value, defs, expanding)
        elif isinstance(value, list):
            result[key] = [
                _resolve_refs(item, defs, expanding) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            result[key] = value

    return result


def _adapt_schema_for_vertex_ai(schema_dict: dict[str, Any]) -> dict[str, Any]:
    """Return a Pydantic JSON schema dict adapted for Vertex AI compatibility.

    *schema_dict* itself is left unmodified; the adapted schema is returned as
    a new dict.

    Two conversions:

    1. ``{"const": "X"}`` -> ``{"enum": ["X"]}``: Vertex AI's proto schema has
       no ``const`` field; a single-value ``enum`` is the equivalent.
    2. ``{"anyOf": [T, {"type": "null"}]}`` -> ``{**T, "nullable": True}``:
       Vertex AI has no ``"null"`` type; optionality is expressed via
       ``"nullable": true`` instead. Also drops ``"default": null`` since
       Vertex AI doesn't need it.
    """

    def _fix(obj: dict[str, Any]) -> None:
        if "const" in obj:
            obj["enum"] = [obj.pop("const")]

        if "anyOf" in obj:
            non_null = [s for s in obj["anyOf"] if s != {"type": "null"}]
            has_null = len(non_null) < len(obj["anyOf"])
            if has_null and len(non_null) == 1:
                obj.update(non_null[0])
                del obj["anyOf"]
                obj["nullable"] = True
            if "default" in obj and obj["default"] is None:
                del obj["default"]

        for v in list(obj.values()):
            if isinstance(v, dict):
                _fix(cast(dict[str, Any], v))
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, dict):
                        _fix(cast(dict[str, Any], item))

    result = deepcopy(schema_dict)
    _fix(result)
    for def_schema in result.get("$defs", {}).values():
        _fix(def_schema)
    return result


# ---------------------------------------------------------------------------
# Formatter interface
# ---------------------------------------------------------------------------


class BatchRequestFormatter(ABC):
    """Builds one provider's batch request line from an ``LLMMessage`` list.

    Subclasses supply the provider-specific request envelope; there is no
    shared logic to keep here beyond the interface itself, since request
    shapes vary enough between providers (Vertex's ``contents``/
    ``system_instruction`` split vs. a forced-tool-call schema elsewhere) that
    factoring out a common implementation would just be indirection.
    """

    #: The :class:`BatchModelParams` subclass this formatter's ``model_params``
    #: accepts. Lets generic code (e.g. something building a formatter from an
    #: existing LLM's ``model_params``) find the right params type for a given
    #: formatter without a hardcoded provider mapping.
    model_params_type: ClassVar[type[BatchModelParams]]

    @abstractmethod
    def format(self, key: str, messages: list[LLMMessage]) -> BatchRequestLine:
        """Render *messages* as the batch request line correlated by *key*.

        Args:
            key: Caller-supplied correlation id, unique within the batch job.
                Not interpreted here; echoed back in the response so a reader
                can join it back to whatever produced *key*.
            messages: The conversation to render.
        """


# ---------------------------------------------------------------------------
# Vertex AI implementation
# ---------------------------------------------------------------------------


class VertexBatchRequestFormatter(BatchRequestFormatter):
    """Renders ``LLMMessage`` lists as Vertex AI Gemini batch prediction request lines.

    Each formatted line follows the `Vertex AI Gemini batch prediction request
    schema <https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/batch-prediction-api>`_::

        {
          "key": "<caller-supplied id>",
          "request": {
            "contents": [{"role": "user", "parts": [{"text": "..."}]}],
            "system_instruction": {"parts": [{"text": "..."}]},
            "generationConfig": {
              "temperature": 0,
              "responseMimeType": "application/json",
              "responseSchema": { ... }
            }
          }
        }

    One formatter instance corresponds to one batch job's generation
    configuration: construct it once with the params and response schema that
    job should use, then call :meth:`format` per request. This mirrors
    :class:`~neo4j_graphrag.llm.batch.client.BaseBatchClient`, whose model is
    likewise fixed at construction rather than passed per call.

    Args:
        model_params: Generation params applied to every formatted request.
            ``None`` (default) uses :class:`VertexModelParams`'s own defaults.
        response_format: JSON schema placed at
            ``generationConfig.responseSchema`` to constrain the model to
            structured output, either as a Pydantic model class (its
            ``model_json_schema()`` is used) or an already-built JSON schema
            dict — the same two shapes :meth:`LLMInterfaceV2.invoke
            <neo4j_graphrag.llm.base.LLMInterfaceV2.invoke>` accepts as
            ``response_format``, so a formatter can be built from the same
            value a caller would otherwise pass to the interactive path.
            Either shape is adapted for Vertex's batch-prediction proto parser
            (see :meth:`_adapt_response_schema`) before being stored.
            ``None`` (default) omits ``responseSchema`` and leaves the model
            free-form.
    """

    model_params_type: ClassVar[type[BatchModelParams]] = VertexModelParams

    # Gemini's content role vocabulary differs from LLMMessage's: it has no
    # "system" role (system content moves to its own top-level field, see
    # `_system_instruction`) and calls the model's own turns "model" rather
    # than "assistant".
    _GEMINI_ROLE_BY_MESSAGE_ROLE: ClassVar[Mapping[str, str]] = {
        "user": "user",
        "assistant": "model",
    }

    def __init__(
        self,
        model_params: VertexModelParams | None = None,
        response_format: type[BaseModel] | dict[str, Any] | None = None,
    ) -> None:
        self._model_params = model_params or VertexModelParams()
        schema = (
            response_format.model_json_schema()
            if isinstance(response_format, type)
            and issubclass(response_format, BaseModel)
            else response_format
        )
        self._response_schema = (
            self._adapt_response_schema(schema) if schema is not None else None
        )

    def format(self, key: str, messages: list[LLMMessage]) -> BatchRequestLine:
        """Render *messages* as the batch request line correlated by *key*.

        Any ``"system"`` messages are collected into ``system_instruction``
        rather than ``contents``, matching how Gemini separates the two.
        """
        request: dict[str, Any] = {"contents": self._contents(messages)}
        system_instruction = self._system_instruction(messages)
        if system_instruction is not None:
            request["system_instruction"] = system_instruction
        request["generationConfig"] = self._generation_config()
        record = {"key": key, "request": request}
        return BatchRequestLine(key=key, line=json.dumps(record, ensure_ascii=False))

    @classmethod
    def _contents(cls, messages: list[LLMMessage]) -> list[dict[str, Any]]:
        return [
            {
                "role": cls._GEMINI_ROLE_BY_MESSAGE_ROLE.get(
                    message["role"], message["role"]
                ),
                "parts": [{"text": message["content"]}],
            }
            for message in messages
            if message["role"] != "system"
        ]

    @staticmethod
    def _system_instruction(messages: list[LLMMessage]) -> dict[str, Any] | None:
        system_texts = [
            message["content"] for message in messages if message["role"] == "system"
        ]
        if not system_texts:
            return None
        return {"parts": [{"text": text} for text in system_texts]}

    def _generation_config(self) -> dict[str, Any]:
        config = self._model_params.to_request_fields()
        if self._response_schema is not None:
            config["responseSchema"] = self._response_schema
        return config

    @classmethod
    def _adapt_response_schema(cls, schema: dict[str, Any]) -> dict[str, Any]:
        """Inline ``$ref``, apply the Vertex conversions, then the batch proto form.

        Neither ``resolve_json_schema_refs`` nor ``_adapt_schema_for_vertex_ai``
        mutates its input, so the caller's schema is never touched.
        """
        resolved = _resolve_json_schema_refs(schema)
        adapted = _adapt_schema_for_vertex_ai(resolved)
        return cast(dict[str, Any], cls._to_batch_proto_schema(adapted))

    @classmethod
    def _to_batch_proto_schema(cls, obj: Any) -> Any:
        """Rewrite an adapted JSON schema into batch-prediction-proto-valid form.

        The interactive ``generateContent`` endpoint accepts raw JSON Schema, but
        the batch-prediction service parses ``responseSchema`` strictly into the
        ``google.cloud.aiplatform`` ``Schema`` proto, which:

          * requires upper-case ``Type`` enum values (``"object"`` -> ``"OBJECT"``),
            and
          * models ``additionalProperties`` as a nested ``Schema``, so the
            closed-object marker ``"additionalProperties": false`` is not a valid
            value and must be dropped.

        Open maps — ``additionalProperties`` whose value is itself a schema — are
        preserved so free-form property bags survive; dropping them too would
        collapse them to an empty object and lose every value.
        """
        if isinstance(obj, dict):
            out: dict[str, Any] = {}
            for key, value in obj.items():
                if key == "type" and isinstance(value, str):
                    out[key] = value.upper()
                elif key == "additionalProperties" and value is False:
                    continue
                else:
                    out[key] = cls._to_batch_proto_schema(value)
            return out
        if isinstance(obj, list):
            return [cls._to_batch_proto_schema(item) for item in obj]
        return obj
