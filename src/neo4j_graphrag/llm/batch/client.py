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
"""Provider-agnostic submit/poll interface for batch inference clients, plus
the Vertex AI implementation.

One batch job looks the same from a workflow's point of view whichever service
runs it: hand over an input file, get back a job handle, poll the handle until
it stops changing.  :class:`BaseBatchClient` is that shape.  Subclasses supply
the transport (:meth:`~BaseBatchClient.submit`,
:meth:`~BaseBatchClient.get`,
:meth:`~BaseBatchClient._job_state_name`); the polling loop, the terminal-state
set and the success comparison live here so every provider agrees on them.
:class:`VertexBatchClient` is the ``google.genai``-backed implementation for
Vertex AI; another transport (e.g. AWS Batch/Bedrock) is a further subclass in
this module.

State names are the Vertex ``JOB_STATE_*`` vocabulary.  That is a historical
accident — Vertex was the first provider wired up — but it is now the shared
vocabulary, so a provider with its own enum maps onto it in
``_job_state_name`` rather than teaching this module a second one.

Building the request file a client submits, and reading the predictions it
writes back, are deliberately not this module's job — see
:mod:`~neo4j_graphrag.llm.batch.formatter` and
:mod:`~neo4j_graphrag.llm.batch.reader`.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

from typing_extensions import Self

try:
    from google import genai
    from google.genai.types import CreateBatchJobConfig, HttpOptions
except ImportError:
    genai = None  # type: ignore[assignment]
    CreateBatchJobConfig = None  # type: ignore[assignment, misc]
    HttpOptions = None  # type: ignore[assignment, misc]

logger = logging.getLogger(__name__)

# The provider-specific job handle a ``get`` returns, threaded through the
# polling loop so a client's ``_job_state_name``/``_job_failure_reason`` are
# typed against the handle their own ``get`` produced.
JobT = TypeVar("JobT")

# The one state that means "done, succeeded".  Callers compare a terminal state
# against this rather than enumerating the failure states.
BATCH_JOB_SUCCEEDED_STATE = "JOB_STATE_SUCCEEDED"

# Default upper bound on total polling wall-time before giving up (24 hours).
# Batch jobs that have not reached a terminal state by then are treated as a
# failure rather than polled forever.
DEFAULT_MAX_WAIT_SECONDS: float = 24 * 60 * 60

# States after which the job will not change further.  Anything outside this set
# (e.g. ``JOB_STATE_RUNNING``/``JOB_STATE_PENDING``, or a ``BATCH_STATE_*`` name
# from an older SDK) is treated as "keep polling" rather than an error.
TERMINAL_BATCH_JOB_STATES: frozenset[str] = frozenset(
    {
        "JOB_STATE_SUCCEEDED",
        "JOB_STATE_FAILED",
        "JOB_STATE_CANCELLED",
        "JOB_STATE_EXPIRED",
    }
)


@dataclass(frozen=True)
class BatchJob:
    """Identifies a submitted batch prediction job.

    Attributes:
        job_name: The provider's own handle for the job, opaque to callers: a
            fully-qualified Vertex AI resource name
            (``projects/my-proj/locations/us-central1/batchPredictionJobs/123``),
            or the inference proxy's ``jobId``, depending on which client
            produced it.  Log or persist it to poll the job later or to
            correlate it with provider-side completion notifications.
    """

    job_name: str


# ---------------------------------------------------------------------------
# Base interface
# ---------------------------------------------------------------------------


class BaseBatchClient(ABC, Generic[JobT]):
    """Shared submit/poll interface for batch prediction clients.

    The model is fixed at construction (from ``LLMConfig.model``).  Subclasses
    implement the provider-specific transport (:meth:`submit`,
    :meth:`get`, :meth:`_job_state_name`); the polling loop in
    :meth:`wait_for` is shared.
    """

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def close(self) -> None:
        """Release any transport the client holds; a no-op by default.

        Declared here so an owner can close whichever
        :class:`BaseBatchClient` subclass it was handed without knowing which
        transport it got.  A client backed by a provider SDK that manages its
        own connections has nothing to release and inherits this; one that
        owns an HTTP or SPIFFE transport overrides it.
        """

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    @abstractmethod
    def submit(
        self,
        requests_uri: str,
        output_bucket_path: str | None = None,
    ) -> BatchJob:
        """Submit a batch prediction job and return its :class:`BatchJob` handle.

        Args:
            requests_uri: Location of the request file the provider reads —
                a ``gs://`` object for Vertex, or an opaque reference to an
                uploaded object for a provider that mints its own upload slots.
            output_bucket_path: Prefix the batch prediction responses are
                written under.  Providers that choose their own output location
                ignore it.
        """

    @abstractmethod
    def get(self, job_name: str) -> JobT:
        """Fetch the current batch job by handle (shape is provider-specific)."""

    @abstractmethod
    def _job_state_name(self, job: JobT) -> str:
        """Extract the job's state name from a :meth:`get` result.

        The returned name is compared against :data:`TERMINAL_BATCH_JOB_STATES`
        and :data:`BATCH_JOB_SUCCEEDED_STATE`, so a provider with its own state
        enum maps onto that vocabulary here.
        """

    def _job_failure_reason(self, job: JobT) -> str | None:
        """Extract why *job* failed, when the provider says.

        Not abstract: a provider that reports nothing beyond a state name keeps
        this default and its callers simply get no reason.

        Returns:
            The provider's explanation, or ``None`` when it gave none or the job
            did not fail.
        """
        return None

    def job_failure_reason(self, job_name: str) -> str | None:
        """Fetch *job_name* and return the provider's reason for its failure.

        A terminal state name on its own ("JOB_STATE_FAILED") says nothing about
        what went wrong, and the reason is the only place a provider-side
        rejection — an over-limit input, an unsupported model — is described at
        all.  Diagnostic output: it is free-form provider text, so log it rather
        than branch on it.

        Costs one extra fetch, so call it only once a job is known to have ended
        badly.

        Returns:
            The provider's explanation, or ``None`` when it gave none.
        """
        return self._job_failure_reason(self.get(job_name))

    def wait_for(
        self,
        job_name: str,
        poll_interval_seconds: float = 60,
        max_wait_seconds: float = DEFAULT_MAX_WAIT_SECONDS,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        fetch: Callable[[], JobT] | None = None,
    ) -> str:
        """Poll *job_name* until it reaches a terminal state and return that state.

        Blocks the calling process, sleeping *poll_interval_seconds* between
        :meth:`get` calls, until the state is in
        :data:`TERMINAL_BATCH_JOB_STATES`.  Non-terminal (and unrecognised) states
        keep the loop running, so an older SDK's ``BATCH_STATE_*`` naming degrades to
        "still running" rather than raising.

        Terminal-but-unsuccessful states are **returned, not raised** — the
        caller decides how each maps onto its own failure handling.

        Gives up after *max_wait_seconds* of total polling so a stuck job never
        blocks the workflow indefinitely.

        Args:
            job_name: The provider's job handle, as carried by :class:`BatchJob`.
            poll_interval_seconds: Seconds to wait between polls.  Defaults to 60.
            max_wait_seconds: Upper bound on total wall-time spent polling before
                raising :class:`TimeoutError`.  Defaults to 24 hours.
            sleep: Sleep function; injected in tests to avoid real waiting.
            monotonic: Monotonic clock used to measure elapsed time; injected in
                tests.  Defaults to :func:`time.monotonic`.
            fetch: Zero-arg fetch returning the current job; injected in tests.
                Defaults to calling :meth:`get` with *job_name*.

        Returns:
            The terminal state name, e.g. ``"JOB_STATE_SUCCEEDED"`` (compare against
            :data:`BATCH_JOB_SUCCEEDED_STATE`).

        Raises:
            TimeoutError: If the job has not reached a terminal state within
                *max_wait_seconds*.
        """
        fetch_job = fetch or (lambda: self.get(job_name))
        start = monotonic()
        while True:
            state = self._job_state_name(fetch_job())
            logger.info("Batch job %s state=%s", job_name, state)
            if state in TERMINAL_BATCH_JOB_STATES:
                return state
            if monotonic() - start >= max_wait_seconds:
                raise TimeoutError(
                    f"Batch job {job_name} did not reach a terminal state within "
                    f"{max_wait_seconds:g}s (last state={state})"
                )
            sleep(poll_interval_seconds)


# ---------------------------------------------------------------------------
# Vertex AI implementation
# ---------------------------------------------------------------------------


class VertexBatchClient(BaseBatchClient["genai.types.BatchJob"]):
    """Submits Vertex AI Gemini batch prediction jobs via the ``google.genai`` client.

    Wraps a lazily-created :class:`google.genai.Client` pointed at Vertex AI.
    The client is constructed on first access to :attr:`client` and reused for
    subsequent submissions.

    Args:
        model_name: The Gemini model to submit against.
        project: GCP project the batch job is billed to and runs in.  ``None``
            lets ``google.genai`` resolve it from ``GOOGLE_CLOUD_PROJECT`` or ADC,
            which is the only option when no Vertex auth is configured.
        location: Vertex AI region the job is submitted to.  ``None`` lets
            ``google.genai`` resolve it from ``GOOGLE_CLOUD_LOCATION``, falling
            back to the ``global`` endpoint.
    """

    def __init__(
        self,
        model_name: str,
        *,
        project: str | None = None,
        location: str | None = None,
    ) -> None:
        if genai is None:
            raise ImportError(
                "Could not import google-genai python client. "
                'Please install it with `pip install "neo4j-graphrag[google-genai]"`.'
            )
        super().__init__(model_name)
        self._project = project
        self._location = location
        self._client: genai.Client | None = None

    @property
    def client(self) -> genai.Client:
        """The underlying Vertex AI ``google.genai`` client, created on first access.

        ``project``/``location`` are passed through as given: ``google.genai``
        reads each as ``value or os.environ[...]``, so ``None`` preserves the
        environment-derived resolution and a configured value overrides it.
        """
        if self._client is None:
            self._client = genai.Client(
                vertexai=True,
                project=self._project,
                location=self._location,
                http_options=HttpOptions(api_version="v1"),
            )
        return self._client

    def submit(
        self,
        requests_uri: str,
        output_bucket_path: str | None = None,
    ) -> BatchJob:
        """Submit a Vertex AI Gemini batch prediction job via the ``google.genai`` client.

        Calls :meth:`client.batches.create <google.genai.batches.Batches.create>`
        with the ``requests.jsonl`` file as the source and *output_bucket_path*
        as the destination prefix.  The job starts asynchronously; this method
        returns as soon as the API accepts the submission.

        Raises:
            google.api_core.exceptions.GoogleAPIError: On Vertex AI API failures
                (invalid model, insufficient quota, etc.).
            RuntimeError: If the API accepted the submission but returned no
                job name — without one the job can never be polled or fetched.
        """
        job = self.client.batches.create(
            model=self.model_name,
            src=requests_uri,
            config=CreateBatchJobConfig(dest=output_bucket_path),
        )
        if not job.name:
            raise RuntimeError(
                "Vertex AI accepted the batch submission but returned no job name"
            )

        logger.info(
            "Submitted Vertex AI batch prediction job: %s (state=%s)",
            job.name,
            job.state,
        )
        return BatchJob(job_name=job.name)

    def get(self, job_name: str) -> genai.types.BatchJob:
        """Fetch the current Vertex AI batch prediction job by resource name.

        Returns:
            The ``google.genai`` batch job object; read ``job.state.name`` for the
            current :class:`JobState`.
        """
        return self.client.batches.get(name=job_name)

    def _job_state_name(self, job: genai.types.BatchJob) -> str:
        """Return ``job.state.name`` defensively (``"UNKNOWN"`` when absent)."""
        return job.state.name if job.state else "UNKNOWN"

    def _job_failure_reason(self, job: genai.types.BatchJob) -> str | None:
        """Render ``job.error``, which Vertex populates only on failure/cancellation.

        ``message`` is the developer-facing text and is what a reader wants;
        ``code`` and ``details`` are included when present because a bare status
        code is still better than nothing when the message is empty.
        """
        error = job.error
        if error is None:
            return None
        parts = [
            part
            for part in (
                f"code={error.code}" if error.code is not None else None,
                error.message,
                "; ".join(error.details) if error.details else None,
            )
            if part
        ]
        return ": ".join(parts) or None
