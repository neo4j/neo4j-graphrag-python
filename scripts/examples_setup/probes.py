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
"""Is a thing present - a package, a command, a database, a model."""

from __future__ import annotations

import json
import shutil
import urllib.error
import urllib.request
from dataclasses import dataclass

import example_requirements as reqs

from .envfile import env_value


def package_installed(module: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


# Extra -> a module that proves it is installed.
EXTRA_PROBE_MODULE: dict[str, str] = {
    "openai": "openai",
    "anthropic": "anthropic",
    "cohere": "cohere",
    "mistralai": "mistralai",
    "ollama": "ollama",
    "bedrock": "boto3",
    "google": "vertexai",
    "google-genai": "google.genai",
    "weaviate": "weaviate",
    "pinecone": "pinecone",
    "qdrant": "qdrant_client",
    "sentence-transformers": "sentence_transformers",
    "nlp": "spacy",
    "fuzzy-matching": "rapidfuzz",
    "kg_creation_tools": "neo4j_viz",
    "experimental": "langchain_text_splitters",
    "examples": "langchain_openai",
    "litellm": "litellm",
}


def extra_installed(extra: str) -> bool:
    module = EXTRA_PROBE_MODULE.get(extra)
    return package_installed(module) if module else True


def command_exists(name: str) -> bool:
    return shutil.which(name) is not None


def ollama_models() -> list[str]:
    """Models already pulled, or [] if the server is not up."""
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2) as r:
            payload = json.loads(r.read().decode())
    except (urllib.error.URLError, OSError, json.JSONDecodeError):
        return []
    models = payload.get("models", [])
    if not isinstance(models, list):
        return []
    return [str(m.get("name", "")) for m in models if isinstance(m, dict)]


@dataclass
class Neo4jState:
    reachable: bool = False
    authenticated: bool = False
    has_apoc: bool = False
    indexes: set[str] = None  # type: ignore[assignment]
    error: str = ""

    def __post_init__(self) -> None:
        if self.indexes is None:
            self.indexes = set()


def probe_neo4j(env_file: dict[str, str]) -> Neo4jState:
    """Inspect the local database. Degrades gracefully before `uv sync`."""
    state = Neo4jState()
    if not reqs.service_available(reqs.NEO4J_LOCAL):
        state.error = "nothing listening on localhost:7687"
        return state
    state.reachable = True

    try:
        import neo4j
    except ImportError:
        state.error = "neo4j driver not installed yet; run uv sync"
        return state

    uri = env_value("NEO4J_URI", env_file) or "bolt://localhost:7687"
    user = env_value("NEO4J_USER", env_file) or "neo4j"
    password = env_value("NEO4J_PASSWORD", env_file) or "password"
    try:
        with neo4j.GraphDatabase.driver(uri, auth=(user, password)) as driver:
            driver.verify_connectivity()
            state.authenticated = True
            records, _, _ = driver.execute_query(
                "SHOW PROCEDURES YIELD name WHERE name STARTS WITH 'apoc' "
                "RETURN count(*) AS c"
            )
            state.has_apoc = bool(records) and records[0]["c"] > 0
            index_records, _, _ = driver.execute_query(
                "SHOW INDEXES YIELD name RETURN name"
            )
            state.indexes = {record["name"] for record in index_records}
    except Exception as exc:  # driver raises a wide range of auth/config errors
        state.error = f"{type(exc).__name__}: {exc}"
    return state
