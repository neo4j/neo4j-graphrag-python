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
"""
Batch inference against Vertex AI Gemini using neo4j_graphrag.llm.batch.

Unlike VertexAILLM.invoke(), which calls the model interactively and waits for
one reply, a batch job runs many prompts asynchronously in one job and can
take anywhere from minutes to hours to finish. This is the offline workflow,
in four steps:

1. Format each prompt as a request line (VertexBatchRequestFormatter).
2. Upload the request file to Cloud Storage: Vertex AI batch prediction reads
   its input from a `gs://` object rather than accepting requests directly.
3. Submit the batch job (VertexBatchClient.submit) and poll it
   (VertexBatchClient.wait_for) until it reaches a terminal state.
4. Read the predictions Vertex wrote back (VertexBatchResponseReader),
   correlating each one to its request by the `key` set in step 1.

Prerequisites:
- A GCP project with the Vertex AI API enabled.
- GOOGLE_APPLICATION_CREDENTIALS set, or running where Application Default
  Credentials otherwise resolve.
- An existing Cloud Storage bucket the caller can read and write.
- pip install "neo4j-graphrag[google-genai]" google-cloud-storage
"""

import tempfile
from pathlib import Path
from typing import Literal

from google.cloud.storage import Client as StorageClient
from pydantic import BaseModel

from neo4j_graphrag.llm.batch import (
    BATCH_JOB_SUCCEEDED_STATE,
    VertexBatchClient,
    VertexBatchRequestFormatter,
    VertexBatchResponseReader,
    VertexModelParams,
)
from neo4j_graphrag.types import LLMMessage

# --- Configuration -----------------------------------------------------------
PROJECT = "example-project"
LOCATION = "us-central1"
MODEL_NAME = "gemini-2.5-flash"
BUCKET = "example-bucket"
# Objects for this run are written under gs://BUCKET/PREFIX/...
PREFIX = "test/neo4j-graphrag/batch-example"


# response_format is translated into Vertex's batch-prediction responseSchema
# (see VertexBatchRequestFormatter._adapt_response_schema). `media_type` is a
# single-value Literal, which Pydantic renders as `{"const": "movie"}`, and
# `tagline` is optional, rendered as `{"anyOf": [{"type": "string"},
# {"type": "null"}]}` — both need Vertex-specific conversions (const -> enum,
# nullable anyOf -> "nullable": true) that have no Vertex proto equivalent
# otherwise. Including both here smoke-tests those conversions against a real
# batch job rather than just unit tests.
class MovieInfo(BaseModel):
    title: str
    year: int
    director: str
    genre: str
    media_type: Literal["movie"] = "movie"
    tagline: str | None = None


# --- 1. Format one request per prompt ----------------------------------------
prompts = {
    "movie-1": "Inception was directed by Christopher Nolan in 2010. "
    "It's a science fiction thriller.",
    "movie-2": "The Godfather was directed by Francis Ford Coppola in 1972. "
    "It's a crime drama.",
}

# temperature=0 here is a generationConfig param applied to every request in
# the job; response_format constrains every prediction to MovieInfo's schema
# (see vertexai_llm_structured_output.py for the interactive equivalent).
formatter = VertexBatchRequestFormatter(
    model_params=VertexModelParams(temperature=0),
    response_format=MovieInfo,
)
request_lines = [
    formatter.format(
        key,
        [
            LLMMessage(
                role="system",
                content="Extract the movie title, year, director and genre as JSON.",
            ),
            LLMMessage(role="user", content=text),
        ],
    )
    for key, text in prompts.items()
]

# --- 2. Upload requests.jsonl to Cloud Storage -------------------------------
storage_client = StorageClient(project=PROJECT)
bucket = storage_client.bucket(BUCKET)

with tempfile.TemporaryDirectory() as tmp_dir:
    requests_path = Path(tmp_dir) / "requests.jsonl"
    requests_path.write_text(
        "\n".join(line.line for line in request_lines) + "\n", encoding="utf-8"
    )
    bucket.blob(f"{PREFIX}/requests.jsonl").upload_from_filename(str(requests_path))

requests_uri = f"gs://{BUCKET}/{PREFIX}/requests.jsonl"
output_prefix = f"gs://{BUCKET}/{PREFIX}/output"

# --- 3. Submit the job and wait for it to finish -----------------------------
client = VertexBatchClient(model_name=MODEL_NAME, project=PROJECT, location=LOCATION)
job = client.submit(requests_uri=requests_uri, output_bucket_path=output_prefix)
print(f"Submitted batch job: {job.job_name}")

# Vertex batch jobs are typically minutes-to-hours; the default
# poll_interval_seconds=60 is fine for a real job, shortened here for the example.
state = client.wait_for(job.job_name, poll_interval_seconds=30)
if state != BATCH_JOB_SUCCEEDED_STATE:
    raise RuntimeError(
        f"Batch job ended in state {state}: {client.job_failure_reason(job.job_name)}"
    )

# --- 4. Read the predictions back --------------------------------------------
# output_bucket_path above is only the prefix Vertex writes under; the exact
# subdirectory (and file names) are only known once the job reports them.
finished_job = client.get(job.job_name)
output_directory = (
    finished_job.output_info.gcs_output_directory if finished_job.output_info else None
)
if output_directory is None:
    raise RuntimeError("Batch job succeeded but reported no output directory")

reader = VertexBatchResponseReader()
output_blob_prefix = output_directory.removeprefix(f"gs://{BUCKET}/")
for blob in bucket.list_blobs(prefix=output_blob_prefix):
    if not blob.name.endswith(".jsonl"):
        continue
    for line in blob.download_as_text().splitlines():
        if not line.strip():
            continue
        record = reader.read(line)
        if record.error:
            print(f"{record.key}: FAILED ({record.error})")
        else:
            print(f"{record.key}: {record.content}")
