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
from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest

from neo4j_graphrag.llm.batch.client import (
    BATCH_JOB_SUCCEEDED_STATE,
    BaseBatchClient,
    BatchJob,
    VertexBatchClient,
)


class StubBatchClient(BaseBatchClient[str]):
    """Minimal concrete client: a job is just its current state name, popped
    from a script on each ``get``."""

    def __init__(self, states: list[str]) -> None:
        super().__init__("test-model")
        self._states = states

    def submit(
        self,
        requests_uri: str,
        output_bucket_path: str | None = None,
    ) -> BatchJob:
        return BatchJob(job_name="stub-job")

    def get(self, job_name: str) -> str:
        return self._states.pop(0)

    def _job_state_name(self, job: str) -> str:
        return job


# ---------------------------------------------------------------------------
# BaseBatchClient.wait_for
# ---------------------------------------------------------------------------


@pytest.fixture
def sleep() -> Iterator[MagicMock]:
    with patch("neo4j_graphrag.llm.batch.client.time.sleep") as mock_sleep:
        yield mock_sleep


def test_wait_for_returns_terminal_state_without_sleeping(sleep: MagicMock) -> None:
    client = StubBatchClient([BATCH_JOB_SUCCEEDED_STATE])

    state = client.wait_for("stub-job")

    assert state == BATCH_JOB_SUCCEEDED_STATE
    sleep.assert_not_called()


def test_wait_for_polls_through_non_terminal_states(sleep: MagicMock) -> None:
    client = StubBatchClient(
        ["JOB_STATE_PENDING", "JOB_STATE_RUNNING", "JOB_STATE_SUCCEEDED"]
    )

    state = client.wait_for("stub-job", poll_interval_seconds=5)

    assert state == "JOB_STATE_SUCCEEDED"
    assert [call.args[0] for call in sleep.call_args_list] == [5, 5]


def test_wait_for_treats_unrecognised_states_as_still_running(
    sleep: MagicMock,
) -> None:
    # An older SDK's BATCH_STATE_* naming degrades to "keep polling" rather
    # than raising.
    client = StubBatchClient(["BATCH_STATE_RUNNING", "JOB_STATE_CANCELLED"])

    state = client.wait_for("stub-job")

    assert state == "JOB_STATE_CANCELLED"


def test_wait_for_returns_terminal_failure_state_without_raising(
    sleep: MagicMock,
) -> None:
    client = StubBatchClient(["JOB_STATE_FAILED"])

    assert client.wait_for("stub-job") == "JOB_STATE_FAILED"


def test_wait_for_times_out_when_never_terminal(sleep: MagicMock) -> None:
    client = StubBatchClient(["JOB_STATE_RUNNING"])

    with (
        patch(
            "neo4j_graphrag.llm.batch.client.time.monotonic",
            side_effect=[0.0, 100.0],
        ),
        pytest.raises(TimeoutError, match="stub-job"),
    ):
        client.wait_for("stub-job", max_wait_seconds=10)


def test_base_job_failure_reason_defaults_to_none() -> None:
    client = StubBatchClient(["JOB_STATE_FAILED"])

    assert client.job_failure_reason("stub-job") is None


# ---------------------------------------------------------------------------
# VertexBatchClient
# ---------------------------------------------------------------------------


@patch("neo4j_graphrag.llm.batch.client.genai", None)
def test_vertex_client_missing_dependency() -> None:
    with pytest.raises(ImportError, match="google-genai"):
        VertexBatchClient(model_name="gemini-2.5-flash")


@patch("neo4j_graphrag.llm.batch.client.genai")
def test_vertex_client_created_lazily_and_reused(mock_genai: MagicMock) -> None:
    client = VertexBatchClient(
        model_name="gemini-2.5-flash", project="my-project", location="eu-west1"
    )
    mock_genai.Client.assert_not_called()

    first = client.client
    second = client.client

    assert first is second
    mock_genai.Client.assert_called_once()
    kwargs = mock_genai.Client.call_args.kwargs
    assert kwargs["vertexai"] is True
    assert kwargs["project"] == "my-project"
    assert kwargs["location"] == "eu-west1"
    assert kwargs["http_options"].api_version == "v1"


@patch("neo4j_graphrag.llm.batch.client.genai")
def test_submit_returns_batch_job_handle(mock_genai: MagicMock) -> None:
    job = MagicMock()
    job.name = "projects/p/locations/l/batchPredictionJobs/1"
    mock_genai.Client.return_value.batches.create.return_value = job
    client = VertexBatchClient(model_name="gemini-2.5-flash", project="p", location="l")

    handle = client.submit("gs://b/requests.jsonl", "gs://b/out")

    assert handle == BatchJob(job_name=job.name)
    create_kwargs = mock_genai.Client.return_value.batches.create.call_args.kwargs
    assert create_kwargs["model"] == "gemini-2.5-flash"
    assert create_kwargs["src"] == "gs://b/requests.jsonl"
    assert create_kwargs["config"].dest == "gs://b/out"


@patch("neo4j_graphrag.llm.batch.client.genai")
def test_submit_raises_when_api_returns_no_job_name(mock_genai: MagicMock) -> None:
    job = MagicMock()
    job.name = None
    mock_genai.Client.return_value.batches.create.return_value = job
    client = VertexBatchClient(model_name="gemini-2.5-flash")

    with pytest.raises(RuntimeError, match="no job name"):
        client.submit("gs://b/requests.jsonl")


@patch("neo4j_graphrag.llm.batch.client.genai")
def test_get_delegates_to_batches_get(mock_genai: MagicMock) -> None:
    client = VertexBatchClient(model_name="gemini-2.5-flash")

    result = client.get("jobs/1")

    mock_genai.Client.return_value.batches.get.assert_called_once_with(name="jobs/1")
    assert result is mock_genai.Client.return_value.batches.get.return_value


def test_job_state_name_returns_unknown_when_state_absent() -> None:
    client = VertexBatchClient(model_name="gemini-2.5-flash")
    job = MagicMock()
    job.state = None

    assert client._job_state_name(job) == "UNKNOWN"


def test_job_failure_reason_includes_code_message_and_details() -> None:
    client = VertexBatchClient(model_name="gemini-2.5-flash")
    job = MagicMock()
    job.error.code = 3
    job.error.message = "invalid argument"
    job.error.details = ["field: src"]

    assert client._job_failure_reason(job) == "code=3: invalid argument: field: src"


def test_job_failure_reason_with_message_only() -> None:
    client = VertexBatchClient(model_name="gemini-2.5-flash")
    job = MagicMock()
    job.error.code = None
    job.error.message = "quota exceeded"
    job.error.details = []

    assert client._job_failure_reason(job) == "quota exceeded"


def test_job_failure_reason_none_when_no_error() -> None:
    client = VertexBatchClient(model_name="gemini-2.5-flash")
    job = MagicMock()
    job.error = None

    assert client._job_failure_reason(job) is None


def test_job_failure_reason_none_when_error_empty() -> None:
    client = VertexBatchClient(model_name="gemini-2.5-flash")
    job = MagicMock()
    job.error.code = None
    job.error.message = ""
    job.error.details = []

    assert client._job_failure_reason(job) is None


def test_job_failure_reason_fetches_the_job_once() -> None:
    client = VertexBatchClient(model_name="gemini-2.5-flash")
    job = MagicMock()
    job.error = None

    with patch.object(client, "get", return_value=job) as mock_get:
        assert client.job_failure_reason("jobs/1") is None

    mock_get.assert_called_once_with("jobs/1")
