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
"""Unit tests for pipeline stage observers."""

from __future__ import annotations

import logging
from typing import Any

import pytest

from neo4j_graphrag.pipeline import (
    Err,
    LocalInterpreter,
    LoggingStageObserver,
    Ok,
    Pipeline,
    StageObserver,
)
from neo4j_graphrag.pipeline.operators import Operator


class _RecordingObserver(StageObserver[Any]):
    """Records every hook invocation as (hook, stage name, payload)."""

    def __init__(self) -> None:
        self.events: list[tuple[str, str, Any]] = []

    def before(self, op: Operator, item: Any) -> None:
        self.events.append(("before", op.name, item))

    def after(self, op: Operator, item: Any) -> None:
        self.events.append(("after", op.name, item))

    def on_error(self, op: Operator, error: Exception) -> None:
        self.events.append(("error", op.name, error))


def _evaluate(pipeline: Pipeline[Any], observer: StageObserver[Any]) -> list[Any]:
    return list(LocalInterpreter(observers=[observer]).evaluate(pipeline))


class TestStageObserverHooks:
    def test_each_item_observed_at_every_stage_boundary(self) -> None:
        observer = _RecordingObserver()
        result = _evaluate(Pipeline([1, 2]).map(lambda x: x * 10), observer)

        assert result == [10, 20]
        assert observer.events == [
            ("before", "SourceOp", 1),
            ("before", "Map", 10),
            ("after", "Map", 10),
            ("after", "SourceOp", 1),
            ("before", "SourceOp", 2),
            ("before", "Map", 20),
            ("after", "Map", 20),
            ("after", "SourceOp", 2),
        ]

    def test_no_hooks_until_stream_consumed(self) -> None:
        observer = _RecordingObserver()
        pipeline = Pipeline([1, 2, 3]).map(lambda x: x + 1)

        stream = LocalInterpreter(observers=[observer]).evaluate(pipeline)
        assert observer.events == []

        assert next(stream) == 2
        assert observer.events == [
            ("before", "SourceOp", 1),
            ("before", "Map", 2),
        ]

    def test_partial_consumption_observes_only_seen_items(self) -> None:
        observer = _RecordingObserver()
        result = _evaluate(Pipeline([1, 2, 3]).take(1), observer)

        assert result == [1]
        # take(1) never resumes the upstream generator after the first
        # item, so "after" hooks for it never run.
        assert [e for e in observer.events if e[1] == "SourceOp"] == [
            ("before", "SourceOp", 1),
        ]

    def test_on_error_fires_then_exception_propagates(self) -> None:
        def explode(x: int) -> int:
            if x == 2:
                raise ValueError("boom")
            return x

        observer = _RecordingObserver()
        stream = LocalInterpreter(observers=[observer]).evaluate(
            Pipeline([1, 2, 3]).map(explode)
        )

        assert next(stream) == 1
        with pytest.raises(ValueError, match="boom"):
            next(stream)

        error_events = [e for e in observer.events if e[0] == "error"]
        assert len(error_events) == 1
        _, stage, error = error_events[0]
        assert stage == "Map"
        assert isinstance(error, ValueError)

    def test_err_values_flow_through_hooks_as_items(self) -> None:
        def fail_on_two(x: int) -> int:
            if x == 2:
                raise ValueError("boom")
            return x

        observer = _RecordingObserver()
        collected = list(
            LocalInterpreter(observers=[observer]).evaluate(
                Pipeline([1, 2]).map_safe(fail_on_two)
            )
        )

        assert isinstance(collected[0], Ok)
        assert isinstance(collected[1], Err)
        err_items = [
            item
            for hook, stage, item in observer.events
            if hook == "before" and isinstance(item, Err)
        ]
        assert len(err_items) == 1

    def test_multiple_observers_all_fire(self) -> None:
        first, second = _RecordingObserver(), _RecordingObserver()
        result = list(
            LocalInterpreter(observers=[first, second]).evaluate(Pipeline([1]).map(str))
        )

        assert result == ["1"]
        assert first.events == second.events != []


class TestLoggingStageObserver:
    def test_logs_stage_boundaries_and_failure(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        def explode(x: int) -> int:
            raise ValueError("boom")

        with caplog.at_level(logging.INFO, logger="neo4j_graphrag.pipeline.observers"):
            list(
                LocalInterpreter(observers=[LoggingStageObserver()]).evaluate(
                    Pipeline([1, 2, 3]).map(lambda x: x * 2)
                )
            )
            with pytest.raises(ValueError, match="boom"):
                list(
                    LocalInterpreter(observers=[LoggingStageObserver()]).evaluate(
                        Pipeline([1]).map(explode)
                    )
                )

        messages = [r.getMessage() for r in caplog.records]
        assert "Stage SourceOp: starting" in messages
        assert "Stage Map: finished, 3 item(s)" in messages
        assert any("Stage Map: failed" in m and "boom" in m for m in messages)
