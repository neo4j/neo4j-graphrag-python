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
"""The services the examples talk to, and how to tell whether they are usable.

Each provider carries the env vars its examples read, what it costs and where to
get a key. Providers whose credentials do not live in an env var - a cloud CLI
login, an AWS named profile - carry a probe instead.
"""

from __future__ import annotations

import subprocess
from functools import cache
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
    signup_url: Optional[str] = None
    free_tier: str = ""
    manual: str = ""
    # For providers whose credentials do not live in an env var (a cloud CLI
    # login, a named profile). Returns (ok, how to fix it).
    credential_probe: Optional[Callable[[], tuple[bool, str]]] = None


@cache
def _probe_aws() -> tuple[bool, str]:
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


@cache
def _probe_gcloud_adc() -> tuple[bool, str]:
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


@cache
def _probe_azure() -> tuple[bool, str]:
    """Azure's example carries no credentials we can detect.

    Its endpoint, key and API version are hardcoded placeholders in the file
    itself rather than read from the environment, so there is nothing to probe -
    and reporting "ready" would send a caller off to run something that cannot
    work. Say so instead.
    """
    return False, (
        "edit the endpoint, key and API version into "
        "examples/customize/embeddings/azure_openai_embeddings.py (do not commit it)"
    )


PROVIDERS: dict[str, Provider] = {
    "openai": Provider(
        key="openai",
        label="OpenAI",
        tier=1,
        env_vars=("OPENAI_API_KEY",),
        signup_url="https://platform.openai.com/api-keys",
        free_tier="No free tier - the key is billed per token.",
    ),
    "gemini": Provider(
        key="gemini",
        label="Google Gemini (AI Studio)",
        tier=1,
        env_vars=("GOOGLE_API_KEY",),
        signup_url="https://aistudio.google.com/apikey",
        free_tier=(
            "Free, no credit card: about 1,500 requests/day and 15 rpm on 2.5 "
            "Flash. Pro models are paid-only. Free-tier prompts may be used to "
            "improve Google's products - do not send anything confidential."
        ),
    ),
    "cohere": Provider(
        key="cohere",
        label="Cohere",
        tier=1,
        env_vars=("CO_API_KEY",),
        signup_url="https://dashboard.cohere.com/api-keys",
        free_tier=(
            "Free trial key, no credit card: 1,000 calls/month, 20 rpm chat and "
            "5 rpm embed. Not licensed for production use."
        ),
    ),
    "mistral": Provider(
        key="mistral",
        label="Mistral AI",
        tier=1,
        env_vars=("MISTRAL_API_KEY",),
        signup_url="https://console.mistral.ai/api-keys",
        free_tier=(
            "Free 'Experiment' tier, no credit card but phone verification is "
            "required. Roughly 1 request/second."
        ),
    ),
    "anthropic": Provider(
        key="anthropic",
        label="Anthropic",
        tier=1,
        env_vars=("ANTHROPIC_API_KEY",),
        signup_url="https://console.anthropic.com/settings/keys",
        free_tier="No free tier - the account needs purchased credits.",
    ),
    "ollama": Provider(
        key="ollama",
        label="Ollama (local)",
        tier=2,
        env_vars=(),
        free_tier="Free and fully local. No account, no key, no network.",
    ),
    "sentence-transformers": Provider(
        key="sentence-transformers",
        label="sentence-transformers (local)",
        tier=2,
        env_vars=(),
        free_tier="Free and local; downloads model weights on first use.",
    ),
    "spacy": Provider(
        key="spacy",
        label="spaCy (local)",
        tier=2,
        env_vars=(),
        free_tier="Free and local; en_core_web_lg is about 560 MB.",
    ),
    "rapidfuzz": Provider(
        key="rapidfuzz",
        label="rapidfuzz (local)",
        tier=2,
        env_vars=(),
        free_tier="Free and local, pure Python dependency.",
    ),
    "weaviate": Provider(
        key="weaviate",
        label="Weaviate",
        tier=2,
        env_vars=(),
        free_tier="Run it locally with Docker - no account needed.",
    ),
    "qdrant": Provider(
        key="qdrant",
        label="Qdrant",
        tier=2,
        env_vars=(),
        free_tier="Run it locally with Docker - no account needed.",
    ),
    "pinecone": Provider(
        key="pinecone",
        label="Pinecone",
        tier=2,
        env_vars=(),
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
        credential_probe=_probe_gcloud_adc,
        free_tier=(
            "No standing free tier; new GCP accounts get $300 in credits. "
            "Gemini via AI Studio reaches the same model family for "
            "free, so only set this up if you specifically need Vertex."
        ),
        manual="gcloud auth application-default login",
    ),
    "bedrock": Provider(
        key="bedrock",
        label="AWS Bedrock",
        tier=3,
        # The examples hardcode region_name, so AWS_REGION is not required.
        # What matters is whether boto3 can resolve credentials.
        env_vars=(),
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
        credential_probe=_probe_azure,
        free_tier="No free tier; needs a deployed Azure OpenAI resource.",
        manual=(
            "examples/customize/embeddings/azure_openai_embeddings.py reads no "
            "env vars - its endpoint and key are hardcoded placeholders you "
            "must edit by hand. Never commit that edit."
        ),
    ),
}

# The models the Ollama examples default to, so a completed setup run leaves
# them runnable with no arguments.
OLLAMA_CHAT_MODEL = "llama3.2"
OLLAMA_EMBED_MODEL = "nomic-embed-text"
