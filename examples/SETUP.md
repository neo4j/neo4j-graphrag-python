# Running the examples

[README.md](README.md) indexes what the examples demonstrate. This file covers what you need
installed and configured to actually run them.

Two commands do most of the work:

```bash
python scripts/setup_examples.py --check   # what is missing, changes nothing
python scripts/setup_examples.py           # walk through fixing it
```

The doctor reports, per example, the single thing standing in the way. The installer works in
tiers, so the free and local providers work before any cloud account is involved.

## Credentials

**Never commit a credential to this repository.** Keys belong in `.env` at the repo root, which is
gitignored. `examples/.env.example` is a template of placeholders and is the only env file that is
tracked.

`setup_examples.py` reads keys with no terminal echo, validates each one against the provider, and
writes it to `.env` with mode `0600`. It never prints a key in full and never writes one to a
tracked file.

Two examples work against this and expect a key edited directly into their source:
`customize/embeddings/azure_openai_embeddings.py`, and the Pinecone examples. If you edit them,
do not commit the edit — use Pinecone Local instead, which ignores keys entirely.

Only 14 examples call `load_dotenv()`. The rest read the environment directly, so export the file
before running one:

```bash
set -a; source .env; set +a
```

## Running an example

Run examples from the repo root — several resolve data files relative to it:

```bash
set -a; source .env; set +a
python examples/question_answering/graphrag.py
```

**Three examples need `examples/data` on the import path.** They load a pre-computed vector as a
bare module (`from embedding_avatar import ...`), which does not resolve from the repo root:

```bash
PYTHONPATH=examples/data python examples/retrieve/similarity_search_for_vector.py
PYTHONPATH=examples/data python examples/customize/retrievers/external/qdrant/qdrant_vector_search.py
PYTHONPATH=examples/data python examples/customize/retrievers/external/weaviate/weaviate_vector_search.py
```

The Ollama examples take the model name as an argument, since it depends on what you have pulled:

```bash
ollama pull llama3.2
python examples/customize/llms/ollama_llm.py llama3.2
```

## Quick start

```bash
uv sync --all-extras --group dev
cp examples/.env.example .env          # then add an OPENAI_API_KEY
docker compose -f examples/docker-compose.yml up -d --wait
python scripts/setup_examples.py --check
```

That covers about two thirds of the examples. For the rest, see the tiers below.

## Services

`examples/docker-compose.yml` starts everything the examples can talk to locally. It uses the same
ports as `tests/e2e/docker-compose.yml`, so **bring one stack down before starting the other**.

```bash
docker compose -f examples/docker-compose.yml up -d --wait                    # Neo4j + APOC
docker compose -f examples/docker-compose.yml --profile vectordb up -d --wait # + vector stores
docker compose -f examples/docker-compose.yml --profile ollama up -d          # + local LLM
```

Sixteen examples instead use the public read-only demo database at `demo.neo4jlabs.com`. Those need
no local setup, just network access.

Some retrievers assume an index already exists. `setup_examples.py --tier 0` creates them
(`moviePlotsEmbedding`, `movieFulltext`, `vector_index`, `fulltext_index`) — including the one
`customize/answer/custom_prompt.py` needs, which no script in the repo previously created.

The vector-store examples need their store populated first:

```bash
uv run python -m tests.e2e.weaviate_e2e.populate_dbs
uv run python -m tests.e2e.qdrant_e2e.populate_dbs
```

(The per-store READMEs under `customize/retrievers/external/` give these as
`python -m tests/e2e/.../populate_dbs.py`, which cannot work — `-m` takes a dotted module path.)

## Providers: what is free, and what has a local equivalent

| Provider | Free without a credit card? | Local / Docker equivalent |
|---|---|---|
| **Google Gemini** (AI Studio) | **Yes.** ~1,500 requests/day on a Flash model. Pro is paid-only, and a new key cannot reach `gemini-2.5-flash` or `gemini-2.0-flash` at all — use `gemini-flash-latest`. Prompts may be used to improve Google's products | — |
| **Cohere** | **Yes.** Trial key: 1,000 calls/month, 20 rpm chat, 5 rpm embed. Not for production | — |
| **Mistral** | **Yes.** "Experiment" tier, ~1 req/s. Phone verification, no card | — |
| **Ollama** | **Yes.** Entirely local | `brew install ollama`, or the `ollama` compose profile |
| **Weaviate / Qdrant** | n/a | Docker, in the `vectordb` profile |
| **Pinecone** | Hosted free "Starter" tier exists | **Pinecone Local** — an in-memory emulator in the `vectordb` profile. Ignores API keys, keeps nothing after it stops |
| **sentence-transformers / spaCy** | n/a | Local model download (spaCy `en_core_web_lg` is ~560 MB) |
| **OpenAI** | No free tier | — |
| **Anthropic** | No free tier; the account needs purchased credits | Claude is also reachable through Bedrock |
| **Vertex AI** | No standing free tier; new GCP accounts get $300 in credits | Gemini via AI Studio reaches the same model family for free |
| **AWS Bedrock** | No standing free tier; accounts created after July 2025 get $200 in credits, expiring after 6 months | — |
| **Azure OpenAI** | No free tier; needs a deployed resource | — |

So every non-OpenAI provider has a free or local path. Tiers 1 and 2 cost nothing.

### Tier 3 notes

- **Vertex AI** authenticates through `gcloud auth application-default login`, not an API key.
  `gcloud auth application-default set-quota-project <project>` is required for
  `VertexAIEmbeddings` but not for `VertexAILLM`.
- **Bedrock** enables serverless models by default, but Anthropic models need a one-time use-case
  form submitted from the Bedrock console before the first call. Credentials come from the standard
  AWS chain — if you use named profiles (SSO commonly writes them), export
  `AWS_PROFILE=<name>` or boto3 will report "Unable to locate credentials" despite a successful
  login.
- If you work at Neo4j, `setup_examples.py --tier 3 --cloud-profile aura` reads your Aura dev
  environment's cloud coordinates live via `omni` instead. Vertex AI is a managed API and works
  while the environment is `SCALED_DOWN`, so there is no need to spin clusters up.

## Python extras

Each provider is behind an extra in `pyproject.toml` — `pip install "neo4j-graphrag[openai]"`, or
`uv sync --all-extras` to get everything at once. `scripts/example_requirements.py` maps every
example to the extras it needs; the doctor reports any that are missing.

Note that examples import library symbols (`OpenAILLM`) rather than provider SDKs, so the
dependency is not visible from an example's imports alone.

## Library bugs that stop examples running

Found by running every example on 2026-08-07. These are defects in `src/neo4j_graphrag/`, not in
your environment — a correct API key does not help.

- **Cohere LLM cannot be constructed.**
  [`cohere_llm.py:106`](../src/neo4j_graphrag/llm/cohere_llm.py#L106) reads
  `cohere.core.api_error.ApiError` as an attribute chain. The top-level `cohere` module resolves
  attributes lazily and does not list `core`, so this raises
  `AttributeError: No core found in _dynamic_imports`. The *module* is present and importable — only
  the attribute access fails — so the fix is one line, `from cohere.core.api_error import ApiError`,
  which works on both the pinned 5.20.1 and the 7.0.8 that renovate wants. Verified on both.
- **Cohere embeddings omit a now-required parameter.** The API rejects the call with
  `invalid request: valid input_type must be provided with the provided model`. Current embed models
  require `input_type`, which the embedder never sends.
- **MistralAI LLM cannot be closed.**
  [`mistralai_llm.py:317`](../src/neo4j_graphrag/llm/mistralai_llm.py#L317) calls
  `self.client.close()`, but `Mistral` has no `close()` in the installed 1.10.0, so any use as a
  context manager ends in `AttributeError: 'Mistral' object has no attribute 'close'`. The
  embeddings class is unaffected.

  **Do not fix this by taking the pending `mistralai` v2 bump.** In 2.9.1 `mistralai` becomes a
  namespace package with no top-level exports, so `from mistralai import Mistral` fails outright —
  the class moved to `mistralai.client`. The v2 upgrade breaks `MistralAILLM` entirely rather than
  fixing it, and needs a real migration.

## Dead model references

- `customize/llms/anthropic_llm.py` uses `claude-3-opus-20240229`, retired. The account's current
  models are `claude-opus-5`, `claude-sonnet-5`, `claude-fable-5` and the 4.x line.
  `anthropic_llm_structured_output.py` is fine — it uses `claude-sonnet-4-5`.
- `customize/llms/google_genai_llm.py` uses `gemini-2.5-flash`, which the API now refuses for new
  keys: *"no longer available to new users"*. **`gemini-flash-latest` works.** Note `gemini-2.0-flash`
  is not an option either — its free-tier quota is `limit: 0`.
- `customize/llms/bedrock_llm.py` uses `us.anthropic.claude-sonnet-4-20250514-v1:0`, which Bedrock
  refuses as *"marked by provider as Legacy"*. `us.anthropic.claude-haiku-4-5-20251001-v1:0` works.
- `customize/llms/vertexai_llm.py` and `vertexai_tool_calls.py` use `gemini-2.0-flash-001`, which
  404s. `VertexAILLM`'s own default, `gemini-1.5-flash-001`, 404s too. `gemini-2.5-flash` works.

## Other known issues in the examples

These are properties of the examples, not of your setup. `--check --verbose` lists them.

- `customize/retrievers/text2cypher_custom_prompt.py` declares `(:User)-[:REVIEWED]->(:Movie)` in
  its schema, which the `recommendations` database does not have.
- `customize/llms/llm_with_neo4j_message_history.py` and
  `question_answering/graphrag_with_neo4j_message_history.py` write message history to the
  **read-only** demo database, so they fail with `Forbidden` as written.
- `customize/build_graph/components/splitters/langhchain_splitter.py` and `llamaindex_splitter.py`
  are empty files, though `README.md` links to them.
- `customize/embeddings/azure_openai_embeddings.py` reads no environment variables at all.
- Three examples import a pre-computed vector from `examples/data/` as a bare module
  (`from embedding_avatar import ...`), which fails from the repo root. Run them with
  `PYTHONPATH=examples/data python <file>`.

## Every example and what it needs

Snippets with no `__main__` and nothing to execute are omitted — they illustrate an API rather than
run. `-` means nothing beyond the core install.

Services: `neo4j-local` is `bolt://localhost:7687`; `neo4j-demo` is the public read-only
`recommendations` database; `apoc` means the example runs APOC procedures.

### Top level

| Example | Provider | Extras | Env vars | Services |
|---|---|---|---|---|
| `kg_builder.py` | openai | openai | - | apoc, neo4j-local |

### build_graph

| Example | Provider | Extras | Env vars | Services |
|---|---|---|---|---|
| `simple_kg_builder_from_pdf.py` | openai | openai | - | apoc, neo4j-local |
| `simple_kg_builder_from_text.py` | openai | openai | - | apoc, neo4j-local |
| `automatic_schema_extraction/simple_kg_builder_schema_from_pdf.py` | openai | examples, openai | NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD | apoc, neo4j-local |
| `automatic_schema_extraction/simple_kg_builder_schema_from_text.py` | openai | examples, openai | NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD | apoc, neo4j-local |
| `from_config_files/simple_kg_pipeline_from_config_file.py` | openai (via config) | openai | NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD | apoc, neo4j-local |
| `from_config_files/simple_kg_pipeline_from_config_file_with_url.py` | openai (via config) | openai | NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD | apoc, neo4j-local, internet |

### customize/answer

| Example | Provider | Extras | Env vars | Services |
|---|---|---|---|---|
| `custom_prompt.py` | openai | openai | - | neo4j-local + `moviePlotsEmbedding` |
| `langchain_compatiblity.py` | openai | examples, openai | - | neo4j-demo |

### customize/build_graph

| Example | Provider | Extras | Env vars | Services |
|---|---|---|---|---|
| `components/chunk_reader/neo4j_chunk_reader.py` | - | - | - | neo4j-local |
| `components/custom_component.py` | - | - | - | - |
| `components/extractors/llm_entity_relation_extractor_with_structured_output.py` | openai, vertexai | examples, google, openai | - | - |
| `components/loaders/pdf_loader.py` | - | - | - | - |
| `components/loaders/pdf_loader_from_url.py` | - | - | - | internet |
| `components/pruners/graph_pruner.py` | - | - | - | - |
| `components/schema_builders/schema_from_existing_graph.py` | - | - | - | neo4j-demo |
| `components/schema_builders/schema_from_text.py` | openai | examples, openai | - | - |
| `components/schema_builders/schema_from_text_with_structured_output.py` | openai, vertexai | examples, google, openai | - | - |
| `pipeline/kg_builder_from_pdf.py` | openai | openai | - | apoc, neo4j-local |
| `pipeline/kg_builder_from_text.py` | openai | openai | - | apoc, neo4j-local |
| `pipeline/kg_builder_two_documents_entity_resolution.py` | openai | openai | - | apoc, neo4j-local |
| `pipeline/lexical_graph_builder_from_text.py` | openai | openai | - | neo4j-local |
| `pipeline/pipeline_streaming.py` | - | - | - | - |
| `pipeline/pipeline_with_component_notifications.py` | - | - | - | - |
| `pipeline/pipeline_with_notifications.py` | - | - | - | neo4j-local |
| `pipeline/text_to_lexical_graph_to_entity_graph_single_pipeline.py` | openai | openai | - | apoc, neo4j-local |
| `pipeline/text_to_lexical_graph_to_entity_graph_two_pipelines.py` | openai | openai | - | apoc, neo4j-local |
| `pipeline/visualization.py` | - | kg_creation_tools | - | - |
| `pipeline/from_config_files/pipeline_from_config_file.py` | - | - | NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD | neo4j-local |

### customize/embeddings

| Example | Provider | Extras | Env vars | Services |
|---|---|---|---|---|
| `azure_openai_embeddings.py` | azure | openai | none — **hardcoded in the file** | - |
| `bedrock_embeddings.py` | bedrock | bedrock | AWS chain, AWS_REGION | - |
| `cohere_embeddings.py` | cohere | cohere | CO_API_KEY | - |
| `custom_embeddings.py` | - | - | - | - |
| `google_genai_embeddings.py` | gemini | google-genai | GOOGLE_API_KEY | - |
| `mistalai_embeddings.py` | mistral | mistralai | MISTRAL_API_KEY | - |
| `ollama_embeddings.py` | ollama | ollama | - | ollama-server |
| `openai_embeddings.py` | openai | openai | OPENAI_API_KEY | - |
| `vertexai_embeddings.py` | vertexai | google | gcloud ADC + quota project | - |

### customize/llms

| Example | Provider | Extras | Env vars | Services |
|---|---|---|---|---|
| `anthropic_llm.py` | anthropic | anthropic | ANTHROPIC_API_KEY | - |
| `anthropic_llm_structured_output.py` | anthropic | anthropic, examples | ANTHROPIC_API_KEY | - |
| `bedrock_llm.py` | bedrock | bedrock | AWS chain, AWS_REGION | - |
| `cohere_llm.py` | cohere | cohere | CO_API_KEY | - |
| `custom_llm.py` | - | - | - | - |
| `google_genai_llm.py` | gemini | google-genai | GOOGLE_API_KEY | - |
| `llm_with_message_history.py` | openai | openai | OPENAI_API_KEY | - |
| `llm_with_neo4j_message_history.py` | openai | openai | OPENAI_API_KEY | neo4j-demo |
| `llm_with_system_instructions.py` | openai | openai | OPENAI_API_KEY | - |
| `mistalai_llm.py` | mistral | mistralai | MISTRAL_API_KEY | - |
| `ollama_llm.py` | ollama | ollama | - | ollama-server |
| `ollama_tool_calls.py` | ollama | ollama | - | ollama-server |
| `openai_llm.py` | openai | openai | OPENAI_API_KEY | - |
| `openai_llm_structured_output.py` | openai | examples, openai | OPENAI_API_KEY | - |
| `openai_tool_calls.py` | openai | examples, openai | OPENAI_API_KEY | - |
| `vertexai_llm.py` | vertexai | google | gcloud ADC | - |
| `vertexai_llm_structured_output.py` | vertexai | google | gcloud ADC | - |
| `vertexai_tool_calls.py` | vertexai | examples, google | gcloud ADC | - |

### customize/retrievers

| Example | Provider | Extras | Env vars | Services |
|---|---|---|---|---|
| `result_formatter_vector_cypher_retriever.py` | openai | openai | OPENAI_API_KEY | neo4j-demo |
| `result_formatter_vector_retriever.py` | openai | openai | OPENAI_API_KEY | neo4j-demo |
| `text2cypher_custom_prompt.py` | openai | openai | OPENAI_API_KEY | neo4j-demo |
| `use_pre_filters.py` | openai | openai | OPENAI_API_KEY | neo4j-demo |
| `hybrid_retrievers/hybrid_cypher_search.py` | - | - | - | neo4j-local |
| `hybrid_retrievers/hybrid_search.py` | - | - | - | neo4j-local |
| `external/pinecone/pinecone_text_search.py` | openai, pinecone | openai, pinecone | OPENAI_API_KEY | neo4j-local, pinecone |
| `external/pinecone/pinecone_vector_search.py` | pinecone, sentence-transformers | pinecone, sentence-transformers | - | neo4j-local, pinecone |
| `external/qdrant/populate_dbs.py` | qdrant | qdrant | - | neo4j-local, qdrant |
| `external/qdrant/qdrant_text_search.py` | qdrant, sentence-transformers | qdrant, sentence-transformers | - | neo4j-local, qdrant |
| `external/qdrant/qdrant_vector_search.py` | qdrant | qdrant | - | neo4j-local, qdrant |
| `external/weaviate/weaviate_text_search_local_embedder.py` | sentence-transformers, weaviate | sentence-transformers, weaviate | - | neo4j-local, weaviate |
| `external/weaviate/weaviate_text_search_remote_embedder.py` | weaviate | weaviate | - | neo4j-local, weaviate |
| `external/weaviate/weaviate_vector_search.py` | weaviate | weaviate | - | neo4j-local, weaviate |

### database_operations

| Example | Provider | Extras | Env vars | Services |
|---|---|---|---|---|
| `create_fulltext_index.py` | - | - | - | neo4j-local |
| `create_vector_index.py` | - | - | - | neo4j-local |
| `populate_vector_index.py` | - | - | - | neo4j-local |

### question_answering

| Example | Provider | Extras | Env vars | Services |
|---|---|---|---|---|
| `graphrag.py` | openai | openai | OPENAI_API_KEY | neo4j-demo |
| `graphrag_with_message_history.py` | openai | openai | OPENAI_API_KEY | neo4j-demo |
| `graphrag_with_neo4j_message_history.py` | openai | openai | OPENAI_API_KEY | neo4j-demo |

### retrieve

| Example | Provider | Extras | Env vars | Services |
|---|---|---|---|---|
| `hybrid_cypher_retriever.py` | openai | openai | OPENAI_API_KEY | neo4j-demo |
| `hybrid_retriever.py` | openai | openai | OPENAI_API_KEY | neo4j-demo |
| `similarity_search_for_text.py` | openai | openai | OPENAI_API_KEY | neo4j-demo |
| `similarity_search_for_vector.py` | - | - | - | neo4j-demo |
| `text2cypher_search.py` | openai | openai | OPENAI_API_KEY | neo4j-demo |
| `vector_cypher_retriever.py` | openai | openai | OPENAI_API_KEY | neo4j-demo |
| `tools/cypher_template_to_tool_example.py` | openai | examples, openai | OPENAI_API_KEY | neo4j-demo |
| `tools/multiple_tools_example.py` | - | - | - | mock driver |
| `tools/retriever_to_tool_example.py` | - | - | - | mock driver |
| `tools/tools_retriever_example.py` | openai | examples, openai | OPENAI_API_KEY | internet (open-meteo) |
