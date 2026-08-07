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
"""What each file under ``examples/`` needs in order to run.

One source of truth, shared by ``check_examples.py`` (which uses it to decide
what ``--live`` may run) and ``setup_examples.py`` (which uses it to report what
is missing and to install it).

Requirements come from three places, in decreasing order of how well they can be
inferred:

1. **Packages.** An example rarely imports a provider SDK directly - it imports
   ``OpenAILLM`` from ``neo4j_graphrag.llm`` and the SDK dependency hides behind
   that. So the import scan maps *library symbols* to extras, not just top-level
   modules, via :data:`SYMBOL_EXTRAS`.
2. **Env vars and datastores.** Read out of the source: ``os.getenv`` calls, and
   connection URIs appearing as string literals.
3. **Everything else** - APOC, a pre-existing index, outbound internet - leaves
   no reliable trace in the source, so it is declared per path in
   :data:`SERVICE_RULES`.

Nothing here imports a third-party package: it must run before ``uv sync``.
"""

from __future__ import annotations

import ast
import fnmatch
import re
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples"

# --------------------------------------------------------------------------
# Providers
# --------------------------------------------------------------------------

# neo4j_graphrag symbols whose use implies an optional dependency. Keyed by the
# symbol as an example would import it; the value is the pyproject extra.
SYMBOL_EXTRAS: dict[str, str] = {
    # LLMs
    "OpenAILLM": "openai",
    "AzureOpenAILLM": "openai",
    "AnthropicLLM": "anthropic",
    "BedrockLLM": "bedrock",
    "CohereLLM": "cohere",
    "GeminiLLM": "google-genai",
    "MistralAILLM": "mistralai",
    "OllamaLLM": "ollama",
    "VertexAILLM": "google",
    # Embedders
    "OpenAIEmbeddings": "openai",
    "AzureOpenAIEmbeddings": "openai",
    "BedrockEmbeddings": "bedrock",
    "CohereEmbeddings": "cohere",
    "GeminiEmbedder": "google-genai",
    "MistralAIEmbeddings": "mistralai",
    "OllamaEmbeddings": "ollama",
    "VertexAIEmbeddings": "google",
    "SentenceTransformerEmbeddings": "sentence-transformers",
    # External retrievers
    "WeaviateNeo4jRetriever": "weaviate",
    "PineconeNeo4jRetriever": "pinecone",
    "QdrantNeo4jRetriever": "qdrant",
    # Components with optional backends
    "SpaCySemanticMatchResolver": "nlp",
    "FuzzyMatchResolver": "fuzzy-matching",
}

# The provider each symbol talks to. Distinct from the extra because one extra
# can front two services (``openai`` covers both OpenAI and Azure OpenAI) and
# one provider can need two extras.
SYMBOL_PROVIDERS: dict[str, str] = {
    "OpenAILLM": "openai",
    "OpenAIEmbeddings": "openai",
    "AzureOpenAILLM": "azure",
    "AzureOpenAIEmbeddings": "azure",
    "AnthropicLLM": "anthropic",
    "BedrockLLM": "bedrock",
    "BedrockEmbeddings": "bedrock",
    "CohereLLM": "cohere",
    "CohereEmbeddings": "cohere",
    "GeminiLLM": "gemini",
    "GeminiEmbedder": "gemini",
    "MistralAILLM": "mistral",
    "MistralAIEmbeddings": "mistral",
    "OllamaLLM": "ollama",
    "OllamaEmbeddings": "ollama",
    "VertexAILLM": "vertexai",
    "VertexAIEmbeddings": "vertexai",
    "SentenceTransformerEmbeddings": "sentence-transformers",
    "WeaviateNeo4jRetriever": "weaviate",
    "PineconeNeo4jRetriever": "pinecone",
    "QdrantNeo4jRetriever": "qdrant",
    "SpaCySemanticMatchResolver": "spacy",
    "FuzzyMatchResolver": "rapidfuzz",
}

# Third-party modules an example may import directly, mapped to their extra.
MODULE_EXTRAS: dict[str, str] = {
    "openai": "openai",
    "anthropic": "anthropic",
    "boto3": "bedrock",
    "botocore": "bedrock",
    "cohere": "cohere",
    "mistralai": "mistralai",
    "ollama": "ollama",
    "google": "google-genai",
    "vertexai": "google",
    "weaviate": "weaviate",
    "pinecone": "pinecone",
    "qdrant_client": "qdrant",
    "sentence_transformers": "sentence-transformers",
    "spacy": "nlp",
    "rapidfuzz": "fuzzy-matching",
    "neo4j_viz": "kg_creation_tools",
    "langchain_openai": "examples",
    "langchain_huggingface": "examples",
    "langchain_text_splitters": "experimental",
    "llama_index": "experimental",
    "pyarrow": "experimental",
    "litellm": "litellm",
    "dotenv": "examples",
}

# Providers reached by importing the module directly rather than via a symbol.
MODULE_PROVIDERS: dict[str, str] = {
    "vertexai": "vertexai",
    "weaviate": "weaviate",
    "pinecone": "pinecone",
    "qdrant_client": "qdrant",
    "sentence_transformers": "sentence-transformers",
    "spacy": "spacy",
    "langchain_openai": "openai",
    "langchain_huggingface": "sentence-transformers",
}

# Import name -> distribution name, where they differ. Used to build install
# hints for packages that are not behind an extra.
DISTRIBUTIONS: dict[str, str] = {
    "dotenv": "python-dotenv",
    "qdrant_client": "qdrant-client",
    "sentence_transformers": "sentence-transformers",
    "neo4j_viz": "neo4j-viz",
}

# --------------------------------------------------------------------------
# Services
# --------------------------------------------------------------------------

NEO4J_LOCAL = "neo4j-local"
NEO4J_DEMO = "neo4j-demo"
APOC = "apoc"
INTERNET = "internet"
OLLAMA_SERVER = "ollama-server"
WEAVIATE_SERVER = "weaviate-server"
QDRANT_SERVER = "qdrant-server"
PINECONE_SERVER = "pinecone-server"

SERVICE_LABELS: dict[str, str] = {
    NEO4J_LOCAL: "local Neo4j on bolt://localhost:7687",
    NEO4J_DEMO: "the demo.neo4jlabs.com recommendations database",
    APOC: "the APOC plugin in the local Neo4j",
    INTERNET: "outbound internet access",
    OLLAMA_SERVER: "a running Ollama server on :11434",
    WEAVIATE_SERVER: "a running Weaviate on :8080",
    QDRANT_SERVER: "a running Qdrant on :6333",
    PINECONE_SERVER: "a running Pinecone (or Pinecone Local on :5080)",
}

# Requirements that leave no reliable trace in the source. Keyed by a glob
# relative to examples/; every matching glob contributes.
#
# `indexes` names a Neo4j index the example expects to already exist. `notes` is
# surfaced verbatim by the doctor - use it for things a user cannot otherwise
# discover without reading the file and failing.
SERVICE_RULES: list[tuple[str, dict[str, object]]] = [
    # Anything that writes an entity graph runs APOC procedures.
    ("kg_builder.py", {"services": [APOC]}),
    ("build_graph/**", {"services": [APOC]}),
    ("customize/build_graph/pipeline/kg_builder_*.py", {"services": [APOC]}),
    (
        "customize/build_graph/pipeline/text_to_lexical_graph_to_entity_graph_*.py",
        {"services": [APOC]},
    ),
    # Fetches its input over the network rather than from examples/data.
    (
        "build_graph/from_config_files/simple_kg_pipeline_from_config_file_with_url.py",
        {"services": [INTERNET]},
    ),
    (
        "customize/build_graph/components/loaders/pdf_loader_from_url.py",
        {"services": [INTERNET]},
    ),
    (
        "retrieve/tools/tools_retriever_example.py",
        {
            "services": [INTERNET],
            "notes": ["calls the live open-meteo.com weather API"],
        },
    ),
    # Indexes the example assumes already exist.
    (
        "retrieve/similarity_search_for_*.py",
        {"indexes": ["moviePlotsEmbedding"]},
    ),
    ("retrieve/vector_cypher_retriever.py", {"indexes": ["moviePlotsEmbedding"]}),
    (
        "retrieve/hybrid_*.py",
        {"indexes": ["moviePlotsEmbedding", "movieFulltext"]},
    ),
    (
        "customize/retrievers/result_formatter_*.py",
        {"indexes": ["moviePlotsEmbedding"]},
    ),
    ("customize/retrievers/use_pre_filters.py", {"indexes": ["moviePlotsEmbedding"]}),
    (
        "customize/answer/custom_prompt.py",
        {
            "indexes": ["moviePlotsEmbedding"],
            "notes": [
                "wants moviePlotsEmbedding on the LOCAL database - no script in "
                "the repo creates it; setup_examples.py --tier 0 does"
            ],
        },
    ),
    (
        "customize/answer/langchain_compatiblity.py",
        {"indexes": ["moviePlotsEmbedding"]},
    ),
    # Vector-store examples need their store populated first.
    (
        "customize/retrievers/external/weaviate/*.py",
        {
            "services": [WEAVIATE_SERVER],
            "notes": ["run tests/e2e/weaviate_e2e/populate_dbs.py before this example"],
        },
    ),
    (
        "customize/retrievers/external/qdrant/*.py",
        {
            "services": [QDRANT_SERVER],
            "notes": ["run tests/e2e/qdrant_e2e/populate_dbs.py before this example"],
        },
    ),
    (
        "customize/retrievers/external/pinecone/*.py",
        {
            "services": [PINECONE_SERVER],
            "notes": [
                "the upstream example expects an API key edited into the source; "
                "prefer Pinecone Local, which ignores keys entirely"
            ],
        },
    ),
    # Known-broken content, flagged rather than fixed. Keeps a user from
    # concluding their own setup is at fault.
    (
        "customize/llms/vertexai_llm.py",
        {"notes": ["model gemini-2.0-flash-001 404s in some projects/regions"]},
    ),
    (
        "customize/llms/vertexai_tool_calls.py",
        {"notes": ["model gemini-2.0-flash-001 404s in some projects/regions"]},
    ),
    (
        "customize/embeddings/azure_openai_embeddings.py",
        {
            "notes": [
                "reads no env vars - endpoint and key are hardcoded placeholders "
                "that must be edited by hand. Never commit the edit"
            ]
        },
    ),
    (
        "customize/llms/llm_with_neo4j_message_history.py",
        {
            "notes": [
                "writes message history to the READ-ONLY demo database, so it "
                "fails with Forbidden as written"
            ]
        },
    ),
    (
        "question_answering/graphrag_with_neo4j_message_history.py",
        {
            "notes": [
                "writes message history to the READ-ONLY demo database, so it "
                "fails with Forbidden as written"
            ]
        },
    ),
    (
        "customize/retrievers/text2cypher_custom_prompt.py",
        {
            "notes": [
                "its schema declares (:User)-[:REVIEWED]->(:Movie), which the "
                "recommendations database does not have"
            ]
        },
    ),
]

# Ollama examples ship a literal placeholder model name; running them needs a
# real model pulled first.
MODEL_PLACEHOLDER = "<model_name>"

_LOCAL_URI = re.compile(r"^(bolt|neo4j)(\+s|\+ssc)?://(localhost|127\.0\.0\.1)")
_DEMO_HOST = "demo.neo4jlabs.com"


@dataclass
class ExampleRequirements:
    """Everything needed to run one example file."""

    path: Path
    extras: set[str] = field(default_factory=set)
    providers: set[str] = field(default_factory=set)
    modules: set[str] = field(default_factory=set)
    env_vars: set[str] = field(default_factory=set)
    services: set[str] = field(default_factory=set)
    indexes: set[str] = field(default_factory=set)
    notes: list[str] = field(default_factory=list)
    runnable: bool = True
    uses_placeholder_model: bool = False
    # Sibling modules under examples/data/ that are importable only if that
    # directory is on sys.path.
    sibling_modules: set[str] = field(default_factory=set)

    @property
    def rel(self) -> str:
        return str(self.path.relative_to(REPO_ROOT))

    def install_hints(self) -> list[str]:
        """Packages this example needs that no extra provides."""
        return sorted(
            DISTRIBUTIONS.get(module, module)
            for module in self.modules
            if module not in MODULE_EXTRAS
        )


def iter_example_files() -> list[Path]:
    return sorted(p for p in EXAMPLES_DIR.rglob("*.py") if "__pycache__" not in p.parts)


def _imported_names(tree: ast.Module) -> tuple[set[str], set[str]]:
    """Return (top-level modules imported, symbols imported by name)."""
    modules: set[str] = set()
    symbols: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative import, never a third-party package
                continue
            if node.module:
                modules.add(node.module.split(".")[0])
            for alias in node.names:
                symbols.add(alias.name)
    return modules, symbols


def _env_vars(tree: ast.Module) -> set[str]:
    """Env var names read via os.getenv / os.environ."""
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _dotted(node.func)
            if name in {"os.getenv", "os.environ.get", "environ.get", "getenv"}:
                if node.args and isinstance(node.args[0], ast.Constant):
                    if isinstance(node.args[0].value, str):
                        found.add(node.args[0].value)
        elif isinstance(node, ast.Subscript):
            if _dotted(node.value) in {"os.environ", "environ"}:
                if isinstance(node.slice, ast.Constant) and isinstance(
                    node.slice.value, str
                ):
                    found.add(node.slice.value)
    return found


def _dotted(node: ast.AST) -> str:
    """Render an attribute chain as ``a.b.c``, or "" if it is not one."""
    parts: list[str] = []
    current: ast.AST = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
        return ".".join(reversed(parts))
    return ""


def _string_constants(tree: ast.Module) -> list[str]:
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]


def _is_main_guard(node: ast.stmt) -> bool:
    if not isinstance(node, ast.If):
        return False
    return any(
        isinstance(child, ast.Constant) and child.value == "__main__"
        for child in ast.walk(node.test)
    )


def _is_runnable(tree: ast.Module) -> bool:
    """Whether the file does something when executed.

    Many examples under ``customize/`` are illustrative snippets - a class or a
    function and nothing else. Running them is a no-op, so the doctor should not
    report them as blocked on credentials they never use.

    Note that plenty of examples have no ``__main__`` guard and simply run at
    module level, several of them inside a top-level ``with`` block, so the guard
    alone is not the test.
    """
    for node in tree.body:
        if _is_main_guard(node):
            return True
        if isinstance(node, (ast.With, ast.AsyncWith, ast.For, ast.While, ast.Try)):
            return True
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            return True
        # A module-level `llm = OpenAILLM(...)` builds a client on import, and
        # `res = llm.invoke(...)` calls out. Both count as running.
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and isinstance(
            node.value, ast.Call
        ):
            return True
    return False


def _sibling_modules(modules: set[str]) -> set[str]:
    """Imports that resolve only against examples/data/.

    A few examples do ``from embedding_avatar import ...``, which fails from the
    repo root because that module lives in examples/data/ rather than beside the
    example. Detected rather than listed, so a new one is caught for free.
    """
    data_dir = EXAMPLES_DIR / "data"
    return {name for name in modules if (data_dir / f"{name}.py").exists()}


def _uses_mock_driver(modules: set[str], symbols: set[str]) -> bool:
    return "MagicMock" in symbols or "unittest" in modules


def _datastore_services(
    literals: list[str], env_vars: set[str], mocked: bool
) -> set[str]:
    if mocked:
        return set()
    services: set[str] = set()
    if any(_DEMO_HOST in value for value in literals):
        services.add(NEO4J_DEMO)
    if any(_LOCAL_URI.match(value) for value in literals):
        services.add(NEO4J_LOCAL)
    # Driven entirely by NEO4J_URI, which every template points at localhost.
    if not services and "NEO4J_URI" in env_vars:
        services.add(NEO4J_LOCAL)
    return services


def _apply_rules(path: Path, requirements: ExampleRequirements) -> None:
    rel = path.relative_to(EXAMPLES_DIR).as_posix()
    for pattern, spec in SERVICE_RULES:
        if not fnmatch.fnmatch(rel, pattern):
            continue
        services = spec.get("services")
        if isinstance(services, list):
            requirements.services.update(str(s) for s in services)
        indexes = spec.get("indexes")
        if isinstance(indexes, list):
            requirements.indexes.update(str(i) for i in indexes)
        notes = spec.get("notes")
        if isinstance(notes, list):
            requirements.notes.extend(str(n) for n in notes)


def analyse(path: Path) -> ExampleRequirements:
    """Work out what one example needs. Never raises on a broken file."""
    requirements = ExampleRequirements(path=path)
    source = path.read_text()
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        requirements.runnable = False
        requirements.notes.append("does not parse")
        return requirements

    modules, symbols = _imported_names(tree)
    requirements.modules = {m for m in modules if m in MODULE_EXTRAS} | {
        m for m in modules if m in DISTRIBUTIONS
    }

    for symbol in symbols:
        if symbol in SYMBOL_EXTRAS:
            requirements.extras.add(SYMBOL_EXTRAS[symbol])
        if symbol in SYMBOL_PROVIDERS:
            requirements.providers.add(SYMBOL_PROVIDERS[symbol])
    for module in modules:
        if module in MODULE_EXTRAS:
            requirements.extras.add(MODULE_EXTRAS[module])
        if module in MODULE_PROVIDERS:
            requirements.providers.add(MODULE_PROVIDERS[module])

    requirements.env_vars = _env_vars(tree)
    requirements.runnable = _is_runnable(tree)
    requirements.uses_placeholder_model = MODEL_PLACEHOLDER in source
    requirements.sibling_modules = _sibling_modules(modules)

    literals = _string_constants(tree)
    mocked = _uses_mock_driver(modules, symbols)
    requirements.services = _datastore_services(literals, requirements.env_vars, mocked)
    if "ollama" in requirements.providers:
        requirements.services.add(OLLAMA_SERVER)

    _apply_rules(path, requirements)

    # APOC and an index are only meaningful alongside a database.
    if not requirements.services & {NEO4J_LOCAL, NEO4J_DEMO}:
        requirements.services.discard(APOC)
    return requirements


def analyse_all() -> list[ExampleRequirements]:
    return [analyse(path) for path in iter_example_files()]


def providers_for_source(path: Path) -> set[str]:
    """Providers one example depends on. Replaces a substring scan.

    ``check_examples.py --live`` uses this to decide what it may run. Matching on
    imported symbols rather than substrings matters: the old scan treated any
    file containing the word "google" as a Vertex example.
    """
    return analyse(path).providers


# --------------------------------------------------------------------------
# Availability probes
# --------------------------------------------------------------------------

# host, port for each service that listens on one.
SERVICE_ENDPOINTS: dict[str, tuple[str, int]] = {
    NEO4J_LOCAL: ("localhost", 7687),
    OLLAMA_SERVER: ("localhost", 11434),
    WEAVIATE_SERVER: ("localhost", 8080),
    QDRANT_SERVER: ("localhost", 6333),
    PINECONE_SERVER: ("localhost", 5080),
}

_PROBE_CACHE: dict[str, bool] = {}


def port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    import socket

    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def service_available(service: str) -> bool:
    """Cheap liveness probe, memoised for the life of the process.

    APOC and index requirements are not probed here - they need an authenticated
    session, so the doctor checks them separately once it has a driver. Treating
    them as available keeps a caller from skipping on something it never tested.
    """
    if service in _PROBE_CACHE:
        return _PROBE_CACHE[service]

    if service == NEO4J_DEMO:
        result = port_open("demo.neo4jlabs.com", 7687, timeout=3.0)
    elif service == INTERNET:
        result = port_open("raw.githubusercontent.com", 443, timeout=3.0)
    elif service in SERVICE_ENDPOINTS:
        host, port = SERVICE_ENDPOINTS[service]
        result = port_open(host, port)
    else:
        result = True

    _PROBE_CACHE[service] = result
    return result
