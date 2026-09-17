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
    ResultPipeline,
    Sink,
    StageObserver,
)
from neo4j_graphrag.pipeline import operators as ops
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


async def _afail(x: int) -> int:
    """Fails for a single item, so a chunk yields one ``Err``."""
    if x == 2:
        raise ValueError("boom")
    return x


class _CaptureSink(Sink[Any]):
    def __init__(self) -> None:
        self.received: list[Any] = []

    def write(self, element: Any) -> None:
        self.received.append(element)


class _FailingSink(Sink[Any]):
    """Records writes until it sees *fail_on*, then raises."""

    def __init__(self, fail_on: Any) -> None:
        self.received: list[Any] = []
        self._fail_on = fail_on

    def write(self, element: Any) -> None:
        if element == self._fail_on:
            raise RuntimeError(f"write failed: {element}")
        self.received.append(element)


def _pipe_tail(pipe: Pipeline[Any]) -> ops.Operator:
    """The tail operator of *pipe*, for building a sink graph by hand."""
    return pipe.pipeline_operators[-1]


def _evaluate(
    pipeline: Pipeline[Any] | ResultPipeline[Any], observer: StageObserver[Any]
) -> list[Any]:
    return list(LocalInterpreter(observers=[observer]).evaluate(pipeline))


class TestStageObserverHooks:
    def test_each_item_observed_at_every_stage_boundary(self) -> None:
        observer = _RecordingObserver()
        result = _evaluate(Pipeline([1, 2]).map(lambda x: x * 10), observer)

        assert result == [10, 20]
        assert observer.events == [
            ("before", "SourceOp[0]", 1),
            ("before", "Map[1]", 10),
            ("after", "Map[1]", 10),
            ("after", "SourceOp[0]", 1),
            ("before", "SourceOp[0]", 2),
            ("before", "Map[1]", 20),
            ("after", "Map[1]", 20),
            ("after", "SourceOp[0]", 2),
        ]

    def test_no_hooks_until_stream_consumed(self) -> None:
        observer = _RecordingObserver()
        pipeline = Pipeline([1, 2, 3]).map(lambda x: x + 1)

        stream = LocalInterpreter(observers=[observer]).evaluate(pipeline)
        assert observer.events == []

        assert next(stream) == 2
        assert observer.events == [
            ("before", "SourceOp[0]", 1),
            ("before", "Map[1]", 2),
        ]

    def test_partial_consumption_observes_only_seen_items(self) -> None:
        observer = _RecordingObserver()
        result = _evaluate(Pipeline([1, 2, 3]).take(1), observer)

        assert result == [1]
        # take(1) never resumes the upstream generator after the first
        # item, so "after" hooks for it never run.
        assert [e for e in observer.events if e[1] == "SourceOp[0]"] == [
            ("before", "SourceOp[0]", 1),
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
        assert stage == "Map[1](explode)"
        assert isinstance(error, ValueError)

    def test_fatal_error_reported_once_by_the_stage_that_raised(self) -> None:
        """A propagating exception is not re-reported by the stages it
        travels through — they did not fail."""

        def explode(x: int) -> int:
            if x == 3:
                raise ValueError("boom")
            return x

        observer = _RecordingObserver()
        stream = LocalInterpreter(observers=[observer]).evaluate(
            Pipeline(range(5)).map(explode).map(lambda v: v + 1).filter(lambda v: True)
        )

        with pytest.raises(ValueError, match="boom"):
            list(stream)

        errors = [
            (stage, item) for hook, stage, item in observer.events if hook == "error"
        ]
        assert len(errors) == 1
        assert errors[0][0] == "Map[1](explode)"

    def test_captured_err_reaches_on_error_once_at_the_capturing_stage(
        self,
    ) -> None:
        def fail_on_two(x: int) -> int:
            if x == 2:
                raise ValueError("boom")
            return x

        observer = _RecordingObserver()
        collected = list(
            LocalInterpreter(observers=[observer]).evaluate(
                Pipeline([1, 2, 3])
                .map_safe(fail_on_two)
                .map_ok(lambda v: v * 10)
                .map_safe(lambda v: v + 1)
            )
        )

        assert [type(item).__name__ for item in collected] == ["Ok", "Err", "Ok"]
        # Reported once, by the stage that captured it — the Err passing
        # through the two later stages is not reported again.
        errors = [
            (stage, item) for hook, stage, item in observer.events if hook == "error"
        ]
        assert len(errors) == 1
        stage, error = errors[0]
        assert stage == "TryMap[1](fail_on_two)"
        assert isinstance(error, ValueError)

    def test_on_error_fires_before_the_err_item_flows(self) -> None:
        def explode(x: int) -> int:
            raise ValueError("boom")

        observer = _RecordingObserver()
        _evaluate(Pipeline([1]).map_safe(explode), observer)

        hooks = [
            hook for hook, stage, _ in observer.events if stage == "TryMap[1](explode)"
        ]
        assert hooks == ["error", "before", "after"]

    @pytest.mark.parametrize(
        ("build", "stage"),
        [
            (
                lambda p: p.map_async_chunked_safe(_afail, map_batch_size=2),
                "TryMapAsyncChunked[1](_afail)",
            ),
            (
                lambda p: p.map_safe(lambda x: x).map_async_chunked_safe(
                    _afail, map_batch_size=2
                ),
                "TryMapOkAsyncChunked[2](_afail)",
            ),
        ],
    )
    def test_async_try_stages_report_captured_errors(
        self, build: Any, stage: str
    ) -> None:
        observer = _RecordingObserver()
        list(
            LocalInterpreter(observers=[observer]).evaluate(build(Pipeline([1, 2, 3])))
        )

        errors = [(s, item) for hook, s, item in observer.events if hook == "error"]
        assert len(errors) == 1
        assert errors[0][0] == stage
        assert isinstance(errors[0][1], ValueError)

    def test_filter_ok_and_on_error_stages_do_not_re_report(self) -> None:
        def fail_on_two(x: int) -> int:
            if x == 2:
                raise ValueError("boom")
            return x

        handled: list[Err] = []
        observer = _RecordingObserver()
        _evaluate(
            Pipeline([1, 2]).map_safe(fail_on_two).on_error(handled.append), observer
        )

        assert len(handled) == 1
        assert len([e for e in observer.events if e[0] == "error"]) == 1

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
        assert "Stage SourceOp[0]: starting" in messages
        assert "Stage Map[1]: finished, 3 item(s)" in messages
        assert any(
            "Stage Map[1](explode): failed" in m and "boom" in m for m in messages
        )


class TestStageNames:
    def test_repeated_stages_are_distinguished_by_position(self) -> None:
        observer = _RecordingObserver()
        _evaluate(
            Pipeline([1]).map(lambda x: x + 1).map(lambda x: x * 2),
            observer,
        )

        assert {stage for _, stage, _ in observer.events} == {
            "SourceOp[0]",
            "Map[1]",
            "Map[2]",
        }

    def test_named_functions_appear_in_the_stage_name(self) -> None:
        def double(x: int) -> int:
            return x * 2

        def is_even(x: int) -> bool:
            return x % 2 == 0

        observer = _RecordingObserver()
        _evaluate(Pipeline([1, 2]).map(double).filter(is_even), observer)

        assert {stage for _, stage, _ in observer.events} == {
            "SourceOp[0]",
            "Map[1](double)",
            "Filter[2](is_even)",
        }

    def test_label_overrides_the_derived_name(self) -> None:
        observer = _RecordingObserver()
        _evaluate(
            Pipeline([1], label="numbers").map(lambda x: x * 2, label="doubling"),
            observer,
        )

        assert {stage for _, stage, _ in observer.events} == {"numbers", "doubling"}


class TestSinkObservation:
    def test_items_written_to_a_sink_are_observed(self) -> None:
        observer = _RecordingObserver()
        sink = _CaptureSink()

        Pipeline([1, 2]).map(lambda x: x * 10).to_sink(
            sink,
            label="capture",
            interpreter=LocalInterpreter(observers=[observer]),
        )

        assert sink.received == [10, 20]
        # The sink's own output stream is empty, so the writes are what
        # gets observed — one before/after pair per item written.
        assert [
            (hook, item) for hook, stage, item in observer.events if stage == "capture"
        ] == [
            ("before", 10),
            ("after", 10),
            ("before", 20),
            ("after", 20),
        ]

    def test_logging_observer_reports_items_written(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.INFO, logger="neo4j_graphrag.pipeline.observers"):
            Pipeline([1, 2, 3]).to_sink(
                _CaptureSink(),
                interpreter=LocalInterpreter(observers=[LoggingStageObserver()]),
            )

        assert "Stage SinkOp[1]: finished, 3 item(s)" in [
            r.getMessage() for r in caplog.records
        ]

    def test_observed_sink_pipeline_still_yields_nothing(self) -> None:
        observer = _RecordingObserver()
        sink = _CaptureSink()
        graph = Pipeline._wrap(
            ops.SinkOp(prev=_pipe_tail(Pipeline([1, 2])), sink=sink, label="capture")
        )

        stream = LocalInterpreter(observers=[observer]).evaluate(graph)
        assert sink.received == []  # evaluate() alone writes nothing
        assert list(stream) == []  # the writes are not passed on to the caller
        assert sink.received == [1, 2]

    def test_a_failing_sink_write_reaches_on_error(self) -> None:
        """The write happens inside the observed stream, so it is attributed."""
        observer = _RecordingObserver()
        sink = _FailingSink(fail_on=2)

        with pytest.raises(RuntimeError, match="write failed: 2"):
            Pipeline([1, 2, 3]).to_sink(
                sink,
                label="capture",
                interpreter=LocalInterpreter(observers=[observer]),
            )

        assert sink.received == [1]
        events = [
            (hook, item) for hook, stage, item in observer.events if stage == "capture"
        ]
        assert events[:2] == [("before", 1), ("after", 1)]
        hook, error = events[2]
        assert hook == "error"
        assert isinstance(error, RuntimeError)
        assert len(events) == 3  # the failed item never reaches before/after


class TestStageObserverBase:
    def test_base_observer_is_usable_and_does_nothing(self) -> None:
        """``StageObserver`` has no abstract methods: all hooks are no-ops."""
        result = _evaluate(Pipeline([1, 2]).map(lambda x: x * 2), StageObserver())

        assert result == [2, 4]


class TestItemRepr:
    def test_long_item_reprs_are_clipped(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        item = "x" * 500

        with caplog.at_level(logging.DEBUG, logger="neo4j_graphrag.pipeline.observers"):
            _evaluate(Pipeline([item]), LoggingStageObserver(max_repr=20))

        emitted = next(
            m for m in (r.getMessage() for r in caplog.records) if "emitting" in m
        )
        assert "[502 chars]" in emitted
        assert len(emitted) < 100

    def test_max_repr_none_logs_the_full_repr(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        item = "x" * 500

        with caplog.at_level(logging.DEBUG, logger="neo4j_graphrag.pipeline.observers"):
            _evaluate(Pipeline([item]), LoggingStageObserver(max_repr=None))

        emitted = next(
            m for m in (r.getMessage() for r in caplog.records) if "emitting" in m
        )
        assert emitted.endswith(repr(item))

    def test_item_repr_not_built_when_debug_is_disabled(self) -> None:
        """An expensive ``__repr__`` is never called if DEBUG is off."""

        class _ExpensiveRepr:
            calls = 0

            def __repr__(self) -> str:
                type(self).calls += 1
                return "expensive"

        log = logging.getLogger("test_observers.quiet")
        log.setLevel(logging.INFO)
        _evaluate(Pipeline([_ExpensiveRepr()]), LoggingStageObserver(log=log))

        assert _ExpensiveRepr.calls == 0
