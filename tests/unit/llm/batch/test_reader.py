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
import json

from neo4j_graphrag.llm.batch.formatter import VertexBatchRequestFormatter
from neo4j_graphrag.llm.batch.reader import (
    BatchResponseRecord,
    VertexBatchResponseReader,
)


def _prediction_line(
    key: str,
    texts: list[str] | None = None,
    status: str = "",
) -> str:
    record: dict[str, object] = {"key": key, "status": status}
    if texts is not None:
        record["response"] = {
            "candidates": [
                {
                    "content": {"parts": [{"text": text} for text in texts]},
                    "role": "model",
                }
            ]
        }
    return json.dumps(record)


def test_read_returns_content_for_successful_row() -> None:
    record = VertexBatchResponseReader().read(_prediction_line("k", ["answer"]))

    assert record == BatchResponseRecord(key="k", content="answer", error=None)


def test_read_joins_multiple_text_parts() -> None:
    record = VertexBatchResponseReader().read(_prediction_line("k", ["a", "b"]))

    assert record.content == "a\nb"


def test_read_skips_parts_without_text() -> None:
    line = json.dumps(
        {
            "key": "k",
            "response": {
                "candidates": [
                    {"content": {"parts": [{"text": "a"}, {}, {"text": "b"}]}}
                ]
            },
        }
    )

    record = VertexBatchResponseReader().read(line)

    assert record.content == "a\nb"


def test_read_returns_error_for_failed_row() -> None:
    # A failed row has a non-empty status and no response; the failure is
    # per-row, so it lands on the record rather than raising.
    record = VertexBatchResponseReader().read(
        _prediction_line("k", texts=None, status="400 Invalid argument")
    )

    assert record == BatchResponseRecord(
        key="k", content=None, error="400 Invalid argument"
    )


def test_read_reports_error_when_response_missing() -> None:
    record = VertexBatchResponseReader().read(_prediction_line("k", texts=None))

    assert record == BatchResponseRecord(
        key="k", content=None, error="missing response"
    )


def test_read_reports_error_when_no_candidates() -> None:
    line = json.dumps({"key": "k", "response": {"candidates": []}})

    record = VertexBatchResponseReader().read(line)

    assert record.content is None
    assert record.error and "no candidates" in record.error


def test_formatted_request_round_trips_through_reader() -> None:
    # The formatter and reader must agree on the envelope: the key a request
    # line carries is the key the prediction line correlates back with.
    request = VertexBatchRequestFormatter().format(
        "req-42", [{"role": "user", "content": "summarise"}]
    )
    prediction_line = _prediction_line(json.loads(request.line)["key"], ["summary"])

    record = VertexBatchResponseReader().read(prediction_line)

    assert record == BatchResponseRecord(key="req-42", content="summary", error=None)
