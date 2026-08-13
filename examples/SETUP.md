# Running the examples

[README.md](README.md) indexes what the examples demonstrate. This file covers what you need
installed and configured to actually run them.

## Quick start

```bash
uv sync --all-extras --group dev
cp examples/.env.example .env          # then add an OPENAI_API_KEY
docker compose -f tests/e2e/docker-compose.yml up -d --wait
set -a; source .env; set +a
python examples/question_answering/graphrag.py
```

That covers about two thirds of the examples. The rest need another provider's key, a vector
store, or a local runtime — see below.

## Credentials

**Never commit a credential to this repository.** Keys belong in `.env` at the repo root, which is
gitignored. `examples/.env.example` is a template of placeholders and is the only env file that is
tracked.

Only 14 examples call `load_dotenv()`. The rest read the environment directly, so export the file
before running one:

```bash
set -a; source .env; set +a
```

Two examples expect a key edited directly into their source rather than read from the
environment: `customize/embeddings/azure_openai_embeddings.py`, and the Pinecone examples. If you
edit them, do not commit the edit — for Pinecone, use Pinecone Local instead, which ignores keys
entirely.

## Running an example

Run examples from the repo root — several resolve data files relative to it.

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

## Services

`tests/e2e/docker-compose.yml` starts everything the examples can talk to locally. The vector
stores sit behind a `vectordb` profile:

```bash
docker compose -f tests/e2e/docker-compose.yml up -d --wait                    # Neo4j + APOC
docker compose -f tests/e2e/docker-compose.yml --profile vectordb up -d --wait # + vector stores
```

17 examples instead use the public read-only demo database at `demo.neo4jlabs.com`. Those need no
local setup, just network access — and because it is read-only, the two examples that write
message history to it fail with `Forbidden`.

Some retriever examples assume a vector or fulltext index already exists on the local database.
`examples/database_operations/` has scripts that create them.

The vector-store examples need their store populated first:

```bash
uv run python -m tests.e2e.weaviate_e2e.populate_dbs
uv run python -m tests.e2e.qdrant_e2e.populate_dbs
```

## Providers: what is free, and what has a local equivalent

| Provider | Free without a credit card? | Local / Docker equivalent |
|---|---|---|
| **Google Gemini** (AI Studio) | **Yes.** Free tier on the Flash models; Pro is paid-only. A new key cannot reach every model the API lists — `gemini-flash-latest` is the safe choice. Prompts may be used to improve Google's products | — |
| **Cohere** | **Yes.** Trial key, rate-limited and not for production | — |
| **Mistral** | **Yes.** "Experiment" tier. Phone verification, no card | — |
| **Ollama** | **Yes.** Entirely local | `brew install ollama` |
| **Weaviate / Qdrant** | n/a | Docker, in the `vectordb` profile |
| **Pinecone** | Hosted free "Starter" tier exists | **Pinecone Local** — an in-memory emulator in the `vectordb` profile. Ignores API keys, keeps nothing after it stops |
| **sentence-transformers / spaCy** | n/a | Local model download (spaCy `en_core_web_lg` is ~560 MB) |
| **OpenAI** | No free tier | — |
| **Anthropic** | No free tier; the account needs purchased credits | Claude is also reachable through Bedrock |
| **Vertex AI** | No standing free tier; new GCP accounts get trial credits | Gemini via AI Studio reaches the same model family for free |
| **AWS Bedrock** | No standing free tier | — |
| **Azure OpenAI** | No free tier; needs a deployed resource | — |

So every non-OpenAI provider has a free or local path.

## Cloud providers

- **Vertex AI** authenticates through `gcloud auth application-default login`, not an API key.
  `gcloud auth application-default set-quota-project <project>` is required for
  `VertexAIEmbeddings` but not for `VertexAILLM`.
- **Bedrock** enables serverless models by default, but Anthropic models need a one-time use-case
  form submitted from the Bedrock console before the first call. Credentials come from the standard
  AWS chain — if you use named profiles (SSO commonly writes them), export `AWS_PROFILE=<name>` or
  boto3 will report "Unable to locate credentials" despite a successful login.
- **Azure OpenAI** needs a deployed resource; the example hardcodes its endpoint and key.

If you work at Neo4j, an Aura dev environment already has these cloud accounts provisioned, and
`omni` will print their coordinates. Vertex AI is a managed API, so it works while the environment
is scaled down.

## Python extras

Each provider is behind an extra in `pyproject.toml` — `pip install "neo4j-graphrag[openai]"`, or
`uv sync --all-extras` to get everything at once.

Note that examples import library symbols (`OpenAILLM`) rather than provider SDKs, so which extra
an example needs is not visible from its imports alone.

## Known issues in the examples

These are properties of the examples, not of your setup.

- `customize/retrievers/text2cypher_custom_prompt.py` declares `(:User)-[:REVIEWED]->(:Movie)` in
  its schema, which the `recommendations` database does not have.
- `customize/llms/llm_with_neo4j_message_history.py` and
  `question_answering/graphrag_with_neo4j_message_history.py` write message history to the
  **read-only** demo database, so they fail with `Forbidden` as written.
- `customize/embeddings/azure_openai_embeddings.py` reads no environment variables. Its endpoint,
  key and API version are placeholders in the file itself (`api_key="<my key>"`), so it has to be
  edited before it will run — and that edit must not be committed.
