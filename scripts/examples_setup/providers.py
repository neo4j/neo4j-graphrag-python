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
"""The services the examples talk to, and how to prove a key works.

Each provider carries what it costs, where to get a key, and a validator. The
validators use a list-models call: free, and it proves the key is live rather
than merely well-formed.
"""

from __future__ import annotations

import subprocess
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Callable, Optional

from .probes import command_exists


@dataclass(frozen=True)
class Provider:
    """A credential-bearing service one or more examples talk to."""

    key: str
    label: str
    tier: int
    env_vars: tuple[str, ...]
    extra: Optional[str] = None
    signup_url: Optional[str] = None
    free_tier: str = ""
    # Returns (ok, detail). Uses a list-models call: free, and it proves the key
    # is live rather than merely well-formed.
    validator: Optional[Callable[[str], tuple[bool, str]]] = None
    manual: str = ""
    # For providers whose credentials do not live in an env var (a cloud CLI
    # login, a named profile). Returns (ok, how to fix it).
    credential_probe: Optional[Callable[[], tuple[bool, str]]] = None


# Credential resolution is the same answer every time within one run, and the
# gcloud probe shells out to mint a token. The doctor asks once per example that
# uses the provider, so without this a single --check spawns it seven times.
_CREDENTIAL_CACHE: dict[str, tuple[bool, str]] = {}


def _cached(key: str, probe: Callable[[], tuple[bool, str]]) -> tuple[bool, str]:
    if key not in _CREDENTIAL_CACHE:
        _CREDENTIAL_CACHE[key] = probe()
    return _CREDENTIAL_CACHE[key]


def _probe_aws() -> tuple[bool, str]:
    return _cached("aws", _probe_aws_uncached)


def _probe_aws_uncached() -> tuple[bool, str]:
    """Whether boto3 can resolve credentials at all.

    A successful SSO login often writes a *named profile*, which boto3 ignores
    unless AWS_PROFILE points at it - so "logged in" and "usable" differ here.
    """
    try:
        import botocore.session
    except ImportError:
        return False, "uv sync --all-extras"
    if botocore.session.get_session().get_credentials() is None:
        return False, "aws sso login, then export AWS_PROFILE=<your profile>"
    return True, ""


def _probe_gcloud_adc() -> tuple[bool, str]:
    return _cached("gcloud-adc", _probe_gcloud_adc_uncached)


def _probe_gcloud_adc_uncached() -> tuple[bool, str]:
    """Whether Vertex can authenticate. The project alone is not enough."""
    if not command_exists("gcloud"):
        return False, "install the gcloud CLI"
    result = subprocess.run(
        ["gcloud", "auth", "application-default", "print-access-token"],
        capture_output=True,
    )
    if result.returncode != 0:
        return False, "gcloud auth application-default login"
    return True, ""


def _http_check(
    url: str, headers: dict[str, str], timeout: float = 15.0
) -> tuple[bool, str]:
    request = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status == 200, f"HTTP {response.status}"
    except urllib.error.HTTPError as exc:
        hint = {
            401: "key rejected",
            403: "key lacks permission",
            429: "rate limited - the key is valid",
        }.get(exc.code, f"HTTP {exc.code}")
        # A rate limit still proves the key authenticates.
        return exc.code == 429, hint
    except urllib.error.URLError as exc:
        # Host only, never the full URL: these strings are printed, and a URL can
        # carry a credential in its query string.
        host = urllib.parse.urlsplit(url).netloc
        return False, f"could not reach {host}: {exc.reason}"
    except OSError as exc:  # pragma: no cover - network shapes vary
        return False, str(exc)


def _validate_openai(key: str) -> tuple[bool, str]:
    return _http_check(
        "https://api.openai.com/v1/models", {"Authorization": f"Bearer {key}"}
    )


def _validate_anthropic(key: str) -> tuple[bool, str]:
    return _http_check(
        "https://api.anthropic.com/v1/models",
        {"x-api-key": key, "anthropic-version": "2023-06-01"},
    )


def _validate_gemini(key: str) -> tuple[bool, str]:
    # Header, not ?key=: a query string is logged by every proxy in the path and
    # by Google's own front end.
    return _http_check(
        "https://generativelanguage.googleapis.com/v1beta/models",
        {"x-goog-api-key": key},
    )


def _validate_cohere(key: str) -> tuple[bool, str]:
    return _http_check(
        "https://api.cohere.com/v1/models", {"Authorization": f"Bearer {key}"}
    )


def _validate_mistral(key: str) -> tuple[bool, str]:
    return _http_check(
        "https://api.mistral.ai/v1/models", {"Authorization": f"Bearer {key}"}
    )


PROVIDERS: dict[str, Provider] = {
    "openai": Provider(
        key="openai",
        label="OpenAI",
        tier=1,
        env_vars=("OPENAI_API_KEY",),
        extra="openai",
        signup_url="https://platform.openai.com/api-keys",
        free_tier="No free tier - the key is billed per token.",
        validator=_validate_openai,
    ),
    "gemini": Provider(
        key="gemini",
        label="Google Gemini (AI Studio)",
        tier=1,
        env_vars=("GOOGLE_API_KEY",),
        extra="google-genai",
        signup_url="https://aistudio.google.com/apikey",
        free_tier=(
            "Free, no credit card: about 1,500 requests/day and 15 rpm on 2.5 "
            "Flash. Pro models are paid-only. Free-tier prompts may be used to "
            "improve Google's products - do not send anything confidential."
        ),
        validator=_validate_gemini,
    ),
    "cohere": Provider(
        key="cohere",
        label="Cohere",
        tier=1,
        env_vars=("CO_API_KEY",),
        extra="cohere",
        signup_url="https://dashboard.cohere.com/api-keys",
        free_tier=(
            "Free trial key, no credit card: 1,000 calls/month, 20 rpm chat and "
            "5 rpm embed. Not licensed for production use."
        ),
        validator=_validate_cohere,
    ),
    "mistral": Provider(
        key="mistral",
        label="Mistral AI",
        tier=1,
        env_vars=("MISTRAL_API_KEY",),
        extra="mistralai",
        signup_url="https://console.mistral.ai/api-keys",
        free_tier=(
            "Free 'Experiment' tier, no credit card but phone verification is "
            "required. Roughly 1 request/second."
        ),
        validator=_validate_mistral,
    ),
    "anthropic": Provider(
        key="anthropic",
        label="Anthropic",
        tier=1,
        env_vars=("ANTHROPIC_API_KEY",),
        extra="anthropic",
        signup_url="https://console.anthropic.com/settings/keys",
        free_tier="No free tier - the account needs purchased credits.",
        validator=_validate_anthropic,
    ),
    "ollama": Provider(
        key="ollama",
        label="Ollama (local)",
        tier=2,
        env_vars=(),
        extra="ollama",
        free_tier="Free and fully local. No account, no key, no network.",
    ),
    "sentence-transformers": Provider(
        key="sentence-transformers",
        label="sentence-transformers (local)",
        tier=2,
        env_vars=(),
        extra="sentence-transformers",
        free_tier="Free and local; downloads model weights on first use.",
    ),
    "spacy": Provider(
        key="spacy",
        label="spaCy (local)",
        tier=2,
        env_vars=(),
        extra="nlp",
        free_tier="Free and local; en_core_web_lg is about 560 MB.",
    ),
    "rapidfuzz": Provider(
        key="rapidfuzz",
        label="rapidfuzz (local)",
        tier=2,
        env_vars=(),
        extra="fuzzy-matching",
        free_tier="Free and local, pure Python dependency.",
    ),
    "weaviate": Provider(
        key="weaviate",
        label="Weaviate",
        tier=2,
        env_vars=(),
        extra="weaviate",
        free_tier="Run it locally with Docker - no account needed.",
    ),
    "qdrant": Provider(
        key="qdrant",
        label="Qdrant",
        tier=2,
        env_vars=(),
        extra="qdrant",
        free_tier="Run it locally with Docker - no account needed.",
    ),
    "pinecone": Provider(
        key="pinecone",
        label="Pinecone",
        tier=2,
        env_vars=(),
        extra="pinecone",
        free_tier=(
            "Pinecone Local is an in-memory Docker emulator that ignores API "
            "keys entirely, so no account is needed. The hosted service also "
            "has a free Starter tier."
        ),
    ),
    "vertexai": Provider(
        key="vertexai",
        label="Google Vertex AI",
        tier=3,
        env_vars=("GOOGLE_CLOUD_PROJECT",),
        extra="google",
        credential_probe=_probe_gcloud_adc,
        free_tier=(
            "No standing free tier; new GCP accounts get $300 in credits. "
            "Gemini via AI Studio (tier 1) reaches the same model family for "
            "free, so only set this up if you specifically need Vertex."
        ),
        manual="Authenticate with gcloud; see tier 3.",
    ),
    "bedrock": Provider(
        key="bedrock",
        label="AWS Bedrock",
        tier=3,
        # The examples hardcode region_name, so AWS_REGION is not required.
        # What matters is whether boto3 can resolve credentials.
        env_vars=(),
        extra="bedrock",
        credential_probe=_probe_aws,
        free_tier=(
            "No standing free tier; accounts created after July 2025 get $200 "
            "in credits that expire after 6 months."
        ),
        manual=(
            "Serverless models are enabled by default, but Anthropic models on "
            "Bedrock need a one-time use-case form submitted from the Bedrock "
            "console before the first call."
        ),
    ),
    "azure": Provider(
        key="azure",
        label="Azure OpenAI",
        tier=3,
        env_vars=(),
        extra="openai",
        free_tier="No free tier; needs a deployed Azure OpenAI resource.",
        manual=(
            "examples/customize/embeddings/azure_openai_embeddings.py reads no "
            "env vars - its endpoint and key are hardcoded placeholders you "
            "must edit by hand. Never commit that edit."
        ),
    ),
}

TIER_NAMES = {
    0: "base - Python extras, local Neo4j, indexes",
    1: "API keys - free tiers first",
    2: "local runtimes - Ollama, local models, vector stores",
    3: "cloud - Vertex AI, Bedrock, Azure OpenAI",
}

# Models pulled for the Ollama examples. qwen3 is currently the most reliable
# small local model for tool calling, which ollama_tool_calls.py needs.
# The models the Ollama examples default to, so a completed setup run leaves
# them runnable with no arguments. Keep in step with examples/SETUP.md.
OLLAMA_CHAT_MODEL = "llama3.2"
OLLAMA_EMBED_MODEL = "nomic-embed-text"

# Indexes the examples expect to already exist on the local database.
LOCAL_INDEXES: list[tuple[str, str, str, int]] = [
    # (name, label, property, dimensions)
    ("moviePlotsEmbedding", "Movie", "plotEmbedding", 1536),
    ("vector_index", "Document", "vectorProperty", 1536),
]
LOCAL_FULLTEXT_INDEXES: list[tuple[str, str, str]] = [
    ("movieFulltext", "Movie", "title"),
    ("fulltext_index", "Document", "textProperty"),
]
