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
import sys
import warnings
from typing import TYPE_CHECKING, Any

# Only the truly provider-agnostic symbols are imported eagerly. The per-provider
# LLM classes import heavy SDKs (``google.cloud.aiplatform``, ``anthropic``,
# ``google.genai``, ...) whose top-level import cost is in the seconds. Eagerly
# importing every provider here forces that cost on every consumer of
# ``neo4j_graphrag.llm`` even when they only use the base interface — a real
# problem for CLI/one-shot workloads that import the package and do one call.
# Provider classes are therefore resolved lazily via :func:`__getattr__`, which
# keeps ``from neo4j_graphrag.llm import BaseLLM`` cheap while preserving
# the public ``from neo4j_graphrag.llm import <Provider>LLM`` API.
from neo4j_graphrag.utils.lazy_import import lazy_dir, lazy_getattr

from .base import BaseLLM, validate_invoke_input
from .types import LLMResponse, LLMUsage
from .utils import split_http_client_kwargs

# Maps each lazily-exported name to the relative submodule it is imported from.
# Kept explicit (rather than deriving a module from the name) so a symbol name
# and its module can diverge, e.g. ``BaseAnthropicLLM`` lives in ``anthropic_llm``.
_LAZY_EXPORTS: dict[str, str] = {
    "AnthropicLLM": ".anthropic_llm",
    "BaseAnthropicLLM": ".anthropic_llm",
    "BedrockLLM": ".bedrock_llm",
    "CohereLLM": ".cohere_llm",
    "GEMINI_DEFAULT_IMAGE_MIME_TYPE": ".google_genai_llm",
    "GEMINI_SUPPORTED_IMAGE_MIME_TYPES": ".google_genai_llm",
    "BaseGeminiLLM": ".google_genai_llm",
    "GeminiImageMimeType": ".google_genai_llm",
    "GeminiLLM": ".google_genai_llm",
    "MistralAILLM": ".mistralai_llm",
    "OllamaLLM": ".ollama_llm",
    "AzureOpenAILLM": ".openai_llm",
    "BaseOpenAILLM": ".openai_llm",
    "OpenAILLM": ".openai_llm",
    "VertexAILLM": ".vertexai_llm",
}

if TYPE_CHECKING:
    # Static-only mirror of _LAZY_EXPORTS so type checkers still resolve the
    # provider classes (and downstream ``# type: ignore`` comments stay
    # meaningful); has no effect at runtime.
    from .anthropic_llm import AnthropicLLM, BaseAnthropicLLM
    from .bedrock_llm import BedrockLLM
    from .cohere_llm import CohereLLM
    from .google_genai_llm import (
        GEMINI_DEFAULT_IMAGE_MIME_TYPE,
        GEMINI_SUPPORTED_IMAGE_MIME_TYPES,
        BaseGeminiLLM,
        GeminiImageMimeType,
        GeminiLLM,
    )
    from .mistralai_llm import MistralAILLM
    from .ollama_llm import OllamaLLM
    from .openai_llm import AzureOpenAILLM, BaseOpenAILLM, OpenAILLM
    from .vertexai_llm import VertexAILLM

__all__ = [
    "GEMINI_DEFAULT_IMAGE_MIME_TYPE",
    "GEMINI_SUPPORTED_IMAGE_MIME_TYPES",
    "AnthropicLLM",
    "BaseAnthropicLLM",
    "BaseGeminiLLM",
    "BedrockLLM",
    "CohereLLM",
    "GeminiImageMimeType",
    "GeminiLLM",
    "LLMResponse",
    "LLMUsage",
    "BaseLLM",
    "OllamaLLM",
    "OpenAILLM",
    "BaseOpenAILLM",
    "VertexAILLM",
    "AzureOpenAILLM",
    "MistralAILLM",
    "split_http_client_kwargs",
    "validate_invoke_input",
]


def __getattr__(name: str) -> Any:
    """Resolve lazily-exported provider classes and deprecated rate-limit names.

    Provider LLM classes are imported on first attribute access so that
    importing ``neo4j_graphrag.llm`` does not pull in every provider SDK
    (see the module docstring).
    """
    # Lazy, per-provider exports first; anything else falls through to the
    # deprecated rate-limit names below.
    if name in _LAZY_EXPORTS:
        return lazy_getattr(name, _LAZY_EXPORTS, sys.modules[__name__])

    from neo4j_graphrag.utils.rate_limit import (
        DEFAULT_RATE_LIMIT_HANDLER,
        NoOpRateLimitHandler,
        RateLimitHandler,
        RetryRateLimitHandler,
        async_rate_limit_handler,
        convert_to_rate_limit_error,
        is_rate_limit_error,
        rate_limit_handler,
    )

    deprecated_items = {
        "RateLimitHandler": RateLimitHandler,
        "NoOpRateLimitHandler": NoOpRateLimitHandler,
        "RetryRateLimitHandler": RetryRateLimitHandler,
        "rate_limit_handler": rate_limit_handler,
        "async_rate_limit_handler": async_rate_limit_handler,
        "is_rate_limit_error": is_rate_limit_error,
        "convert_to_rate_limit_error": convert_to_rate_limit_error,
        "DEFAULT_RATE_LIMIT_HANDLER": DEFAULT_RATE_LIMIT_HANDLER,
    }

    if name in deprecated_items:
        warnings.warn(
            f"{name} has been moved to neo4j_graphrag.utils.rate_limit. "
            f"Please update your imports to use 'from neo4j_graphrag.utils.rate_limit import {name}'. "
            "This import will be removed in version 2.0",
            DeprecationWarning,
            stacklevel=2,
        )
        return deprecated_items[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Include the lazily-exported names in ``dir(neo4j_graphrag.llm)``."""
    return lazy_dir(_LAZY_EXPORTS, sys.modules[__name__])
