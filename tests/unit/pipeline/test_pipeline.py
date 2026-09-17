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
"""Unit tests for the embedded pipeline DSL.

Tests assert on actual data transformations, not internal structure.
Async operator tests are sync — blocking happens inside the interpreter
via ``asyncio.run()``, so no running event loop is present during the test.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pytest

from neo4j_graphrag.pipeline import (
    Err,
    LocalInterpreter,
    Ok,
    Pipeline,
    ResultPipeline,
    Sink,
    Source,
)
from neo4j_graphrag.pipeline import operators as ops


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class _InMemorySource(Source[int]):
    def __init__(self, items: list[int]) -> None:
        self._items = items

    def read(self) -> Iterable[int]:
        return iter(self._items)


class _CountingSource(Source[int]):
    """Records how many times read() is called."""

    def __init__(self, items: list[int]) -> None:
        self._items = items
        self.reads = 0

    def read(self) -> Iterable[int]:
        self.reads += 1
        return iter(self._items)


class _CaptureSink(Sink[int]):
    def __init__(self) -> None:
        self.received: list[int] = []

    def write(self, element: int) -> None:
        self.received.append(element)


class _FailingSink(Sink[int]):
    """Records writes until it sees *fail_on*, then raises."""

    def __init__(self, fail_on: int) -> None:
        self.received: list[int] = []
        self._fail_on = fail_on

    def write(self, element: int) -> None:
        if element == self._fail_on:
            raise RuntimeError(f"write failed: {element}")
        self.received.append(element)


def pipe_tail(pipe: Pipeline[int]) -> ops.Operator:
    """The tail operator of *pipe*, for building a sink graph by hand."""
    return pipe.pipeline_operators[-1]


async def _double(x: int) -> int:
    return x * 2


async def _fail_on_even(x: int) -> int:
    if x % 2 == 0:
        raise ValueError(f"even: {x}")
    return x


async def _duplicate(x: int) -> list[int]:
    return [x, x * 10]


# ---------------------------------------------------------------------------
# Construction / from_source
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_emits_elements_from_source(self) -> None:
        result = Pipeline.from_source(_InMemorySource([1, 2, 3])).collect()
        assert result == [1, 2, 3]

    def test_empty_source_produces_empty_stream(self) -> None:
        assert Pipeline.from_source(_InMemorySource([])).collect() == []

    def test_source_subclass_satisfied(self) -> None:
        assert isinstance(_InMemorySource([1]), Source)

    def test_source_is_abstract(self) -> None:
        with pytest.raises(TypeError):
            Source()  # type: ignore[abstract]

    def test_wraps_iterable_directly(self) -> None:
        assert Pipeline([1, 2, 3]).collect() == [1, 2, 3]

    def test_iterable_directly(self) -> None:
        pipe = Pipeline([1, 2, 3]).map(lambda x: x * 10)
        assert list(pipe) == [10, 20, 30]

    def test_definition_is_lazy(self) -> None:
        """Building a pipeline executes nothing — not even source.read()."""
        source = _CountingSource([1, 2, 3])
        pipe = Pipeline.from_source(source).map(lambda x: x + 1)
        assert source.reads == 0
        assert pipe.collect() == [2, 3, 4]
        assert source.reads == 1

    def test_explicit_interpreter(self) -> None:
        pipe = Pipeline([1, 2, 3]).map(lambda x: x + 1)
        assert list(LocalInterpreter().evaluate(pipe)) == [2, 3, 4]

    def test_pipeline_operators_in_evaluation_order(self) -> None:
        pipe = Pipeline([1]).map(lambda x: x).filter(lambda x: True)
        kinds = [type(op) for op in pipe.pipeline_operators]
        assert kinds == [ops.SourceOp, ops.Map, ops.Filter]

    def test_reiterable_source_can_be_evaluated_twice(self) -> None:
        pipe = Pipeline([1, 2, 3]).map(lambda x: x + 1)
        assert pipe.collect() == [2, 3, 4]
        assert pipe.collect() == [2, 3, 4]


# ---------------------------------------------------------------------------
# map
# ---------------------------------------------------------------------------


class TestMap:
    def test_transforms_each_element(self) -> None:
        assert Pipeline([1, 2, 3]).map(lambda x: x * 2).collect() == [2, 4, 6]

    def test_changes_element_type(self) -> None:
        assert Pipeline([1, 2]).map(str).collect() == ["1", "2"]

    def test_empty_stream(self) -> None:
        assert Pipeline([]).map(lambda x: x).collect() == []

    def test_chained_maps_compose(self) -> None:
        result = Pipeline([1]).map(lambda x: x + 1).map(lambda x: x * 10).collect()
        assert result == [20]

    def test_exception_propagates(self) -> None:
        def _boom(x: int) -> int:
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            Pipeline([1]).map(_boom).collect()


# ---------------------------------------------------------------------------
# flat_map
# ---------------------------------------------------------------------------


class TestFlatMap:
    def test_flattens_one_level(self) -> None:
        result = Pipeline([1, 2, 3]).flat_map(lambda x: [x, x * 10]).collect()
        assert result == [1, 10, 2, 20, 3, 30]

    def test_empty_inner_iterables_are_dropped(self) -> None:
        result = Pipeline([1, 2, 3]).flat_map(lambda x: [] if x == 2 else [x]).collect()
        assert result == [1, 3]

    def test_empty_stream(self) -> None:
        assert Pipeline([]).flat_map(lambda x: [x]).collect() == []


# ---------------------------------------------------------------------------
# take / take_while / skip
# ---------------------------------------------------------------------------


class TestTake:
    def test_takes_first_n_elements(self) -> None:
        assert Pipeline([1, 2, 3, 4, 5]).take(3).collect() == [1, 2, 3]

    def test_empty_stream(self) -> None:
        assert Pipeline([]).take(3).collect() == []

    def test_negative_n_raises(self) -> None:
        with pytest.raises(ValueError, match="n must be >= 0"):
            Pipeline([1, 2, 3]).take(-1)

    def test_n_larger_than_stream(self) -> None:
        assert Pipeline([1, 2, 3]).take(4).collect() == [1, 2, 3]

    def test_n_zero(self) -> None:
        assert Pipeline([1, 2, 3]).take(0).collect() == []


class TestTakeWhile:
    def test_takes_elements_while_predicate_is_true(self) -> None:
        result = Pipeline([1, 2, 3, 1]).take_while(lambda x: x < 3).collect()
        assert result == [1, 2]

    def test_stops_at_first_false_not_filter(self) -> None:
        """take_while stops; it does not skip non-matching elements."""
        result = Pipeline([2, 4, 3, 6]).take_while(lambda x: x % 2 == 0).collect()
        assert result == [2, 4]

    def test_predicate_false_on_first_element_returns_empty(self) -> None:
        assert Pipeline([5, 1]).take_while(lambda x: x < 3).collect() == []

    def test_predicate_always_true_returns_all(self) -> None:
        assert Pipeline([1, 2]).take_while(lambda x: True).collect() == [1, 2]


class TestSkip:
    def test_skips_first_n_elements(self) -> None:
        assert Pipeline([1, 2, 3, 4]).skip(2).collect() == [3, 4]

    def test_negative_n_raises(self) -> None:
        with pytest.raises(ValueError, match="n must be >= 0"):
            Pipeline([1]).skip(-1)

    def test_n_larger_than_stream(self) -> None:
        assert Pipeline([1, 2]).skip(5).collect() == []


# ---------------------------------------------------------------------------
# filter
# ---------------------------------------------------------------------------


class TestFilter:
    def test_keeps_matching_elements(self) -> None:
        result = Pipeline([1, 2, 3, 4, 5]).filter(lambda x: x % 2 == 0).collect()
        assert result == [2, 4]

    def test_all_filtered_out(self) -> None:
        assert Pipeline([1, 3]).filter(lambda x: x % 2 == 0).collect() == []

    def test_none_filtered_out(self) -> None:
        assert Pipeline([2, 4]).filter(lambda x: x % 2 == 0).collect() == [2, 4]


# ---------------------------------------------------------------------------
# grouped
# ---------------------------------------------------------------------------


class TestGrouped:
    def test_splits_into_full_batches(self) -> None:
        result = Pipeline([1, 2, 3, 4, 5, 6]).grouped(2).collect()
        assert result == [[1, 2], [3, 4], [5, 6]]

    def test_final_batch_may_be_smaller(self) -> None:
        result = Pipeline([1, 2, 3, 4, 5]).grouped(2).collect()
        assert result == [[1, 2], [3, 4], [5]]

    def test_size_larger_than_stream(self) -> None:
        assert Pipeline([1, 2]).grouped(10).collect() == [[1, 2]]

    def test_size_one(self) -> None:
        assert Pipeline([1, 2]).grouped(1).collect() == [[1], [2]]

    def test_empty_stream(self) -> None:
        assert Pipeline([]).grouped(3).collect() == []

    def test_size_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="grouped size must be >= 1"):
            Pipeline([1]).grouped(0)

    def test_negative_size_raises(self) -> None:
        with pytest.raises(ValueError, match="grouped size must be >= 1"):
            Pipeline([1]).grouped(-2)


# ---------------------------------------------------------------------------
# reduce
# ---------------------------------------------------------------------------


class TestReduce:
    def test_sums_elements(self) -> None:
        result = Pipeline([1, 2, 3, 4]).reduce(0, lambda a, x: a + x).collect()
        assert result == [10]

    def test_identity_on_empty_stream(self) -> None:
        result = Pipeline([]).reduce(0, lambda a, x: a + x).collect()
        assert result == [0]

    def test_string_concatenation(self) -> None:
        result = Pipeline(["a", "b", "c"]).reduce("", lambda a, x: a + x).collect()
        assert result == ["abc"]

    def test_result_can_be_mapped(self) -> None:
        result = (
            Pipeline([1, 2, 3])
            .reduce(0, lambda a, x: a + x)
            .map(lambda total: total * 10)
            .collect()
        )
        assert result == [60]


# ---------------------------------------------------------------------------
# map_async_chunked
# ---------------------------------------------------------------------------


class TestMapAsyncChunked:
    def test_transforms_all_elements(self) -> None:
        result = Pipeline([1, 2, 3]).map_async_chunked(_double).collect()
        assert result == [2, 4, 6]

    def test_empty_stream_collects_empty_list(self) -> None:
        assert Pipeline([]).map_async_chunked(_double).collect() == []

    def test_chunk_size_larger_than_stream(self) -> None:
        result = (
            Pipeline([1, 2]).map_async_chunked(_double, map_batch_size=100).collect()
        )
        assert result == [2, 4]

    def test_map_batch_size_zero_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="map_batch_size must be >= 1"):
            Pipeline([1]).map_async_chunked(_double, map_batch_size=0)

    def test_results_yielded_per_chunk_in_order(self) -> None:
        result = (
            Pipeline([1, 2, 3, 4, 5])
            .map_async_chunked(_double, map_batch_size=2)
            .collect()
        )
        assert result == [2, 4, 6, 8, 10]


# ---------------------------------------------------------------------------
# to_sink / collect
# ---------------------------------------------------------------------------


class TestToSink:
    def test_writes_all_elements_to_sink(self) -> None:
        sink = _CaptureSink()
        Pipeline([1, 2, 3]).to_sink(sink)
        assert sink.received == [1, 2, 3]

    def test_empty_stream_writes_nothing(self) -> None:
        sink = _CaptureSink()
        Pipeline([]).to_sink(sink)
        assert sink.received == []

    def test_sink_subclass_satisfied(self) -> None:
        assert isinstance(_CaptureSink(), Sink)

    def test_sink_is_abstract(self) -> None:
        with pytest.raises(TypeError):
            Sink()  # type: ignore[abstract]

    def test_chained_operators_apply_before_sink(self) -> None:
        sink = _CaptureSink()
        Pipeline([1, 2, 3]).map(lambda x: x * 2).to_sink(sink)
        assert sink.received == [2, 4, 6]

    def test_write_failure_aborts_the_run(self) -> None:
        """Sink failures are fatal — they are not captured as ``Err``."""
        sink = _FailingSink(fail_on=2)
        with pytest.raises(RuntimeError, match="write failed: 2"):
            Pipeline([1, 2, 3]).to_sink(sink)

    def test_elements_before_a_write_failure_stay_written(self) -> None:
        sink = _FailingSink(fail_on=2)
        with pytest.raises(RuntimeError):
            Pipeline([1, 2, 3]).to_sink(sink)
        assert sink.received == [1]

    def test_write_failure_stops_reading_the_upstream(self) -> None:
        """The failure must not drain the rest of the source first."""
        seen: list[int] = []

        def _record(x: int) -> int:
            seen.append(x)
            return x

        with pytest.raises(RuntimeError):
            Pipeline([1, 2, 3]).map(_record).to_sink(_FailingSink(fail_on=2))
        assert seen == [1, 2]

    def test_sink_after_chunked_async_stage(self) -> None:
        sink = _CaptureSink()
        Pipeline([1, 2, 3]).map_async_chunked(_double, map_batch_size=2).to_sink(sink)
        assert sink.received == [2, 4, 6]

    def test_sink_receives_result_values_when_errors_are_handled(self) -> None:
        sink = _CaptureSink()
        errors: list[Err] = []
        Pipeline([1, 2, 3]).map_safe(_fail_on_two_sync).on_error(errors.append).to_sink(
            sink
        )
        assert sink.received == [1, 3]
        assert len(errors) == 1


class TestLaziness:
    def test_operators_do_not_execute_until_consumed(self) -> None:
        calls: list[int] = []

        def _record(x: int) -> int:
            calls.append(x)
            return x

        pipe = Pipeline([1, 2, 3]).map(_record)
        assert calls == []  # nothing has run yet
        pipe.collect()
        assert calls == [1, 2, 3]  # runs on consumption

    def test_evaluate_runs_nothing_before_the_stream_is_consumed(self) -> None:
        """``evaluate()`` only builds the generator chain.

        Every operator must defer its work, including the ones that drain
        their whole upstream (``reduce``, ``to_sink``).
        """
        source = _CountingSource([1, 2, 3])
        pipe = Pipeline.from_source(source).map(lambda x: x)
        LocalInterpreter().evaluate(pipe)
        assert source.reads == 0

    def test_reduce_does_not_fold_until_consumed(self) -> None:
        read: list[int] = []

        def _tracked() -> Iterable[int]:
            for i in [1, 2, 3]:
                read.append(i)
                yield i

        pipe = Pipeline(_tracked()).reduce(zero=0, combine=lambda a, b: a + b)
        stream = LocalInterpreter().evaluate(pipe)
        assert read == []  # the fold has not touched the upstream yet
        assert list(stream) == [6]
        assert read == [1, 2, 3]

    def test_sink_does_not_write_until_consumed(self) -> None:
        sink = _CaptureSink()
        base = Pipeline([1, 2, 3])
        sink_graph = Pipeline._wrap(ops.SinkOp(prev=pipe_tail(base), sink=sink))
        stream = LocalInterpreter().evaluate(sink_graph)
        assert sink.received == []  # evaluate() alone writes nothing
        assert list(stream) == []  # a sink stream yields nothing...
        assert sink.received == [1, 2, 3]  # ...but writes as it drains

    def test_to_sink_drains_the_stream(self) -> None:
        sink = _CaptureSink()
        Pipeline([1, 2, 3]).to_sink(sink)
        assert sink.received == [1, 2, 3]

    def test_full_pipeline(self) -> None:
        result = (
            Pipeline([1, 2, 3, 4, 5, 6, 7, 8])
            .filter(lambda x: x % 2 == 0)  # [2, 4, 6, 8]
            .map(lambda x: x * 3)  # [6, 12, 18, 24]
            .flat_map(lambda x: [x, x + 1])  # [6,7,12,13,18,19,24,25]
            .grouped(4)  # [[6,7,12,13], [18,19,24,25]]
            .map(sum)  # [38, 86]
            .reduce(0, lambda a, x: a + x)  # [124]
            .collect()
        )
        assert result == [124]


# ---------------------------------------------------------------------------
# map_safe (Pipeline -> ResultPipeline)
# ---------------------------------------------------------------------------


class TestMapSafe:
    def test_all_succeed_wraps_in_ok(self) -> None:
        result = Pipeline([1, 2]).map_safe(lambda x: x * 2).collect()
        assert result == [Ok(2), Ok(4)]

    def test_exception_becomes_err(self) -> None:
        def _boom(x: int) -> int:
            raise ValueError(f"bad {x}")

        result = Pipeline([1]).map_safe(_boom).collect()
        assert len(result) == 1
        assert isinstance(result[0], Err)
        assert isinstance(result[0].exception, ValueError)

    def test_failure_does_not_abort_stream(self) -> None:
        def _fail_on_two(x: int) -> int:
            if x == 2:
                raise ValueError("two")
            return x

        result = Pipeline([1, 2, 3]).map_safe(_fail_on_two).collect()
        assert result[0] == Ok(1)
        assert isinstance(result[1], Err)
        assert result[2] == Ok(3)

    def test_empty_stream(self) -> None:
        assert Pipeline([]).map_safe(lambda x: x).collect() == []

    def test_returns_result_pipeline(self) -> None:
        assert isinstance(Pipeline([1]).map_safe(lambda x: x), ResultPipeline)


# ---------------------------------------------------------------------------
# flatten_ok — the 1-to-many stage for result streams
# ---------------------------------------------------------------------------


class TestFlattenOk:
    def test_expands_each_ok_into_one_ok_per_item(self) -> None:
        result = Pipeline([1, 2]).map_safe(lambda x: [x, x * 10]).flatten_ok().collect()
        assert result == [Ok(1), Ok(10), Ok(2), Ok(20)]

    def test_err_passes_through_at_original_position(self) -> None:
        result = _mixed_stream().map_safe(lambda x: [x, x]).flatten_ok().collect()
        assert result[0] == Ok(1)
        assert result[1] == Ok(1)
        assert isinstance(result[2], Err)
        assert result[3] == Ok(3)
        assert result[4] == Ok(3)

    def test_upstream_exception_stays_a_single_err(self) -> None:
        def _boom(x: int) -> list[int]:
            raise ValueError("boom")

        result = Pipeline([1]).map_safe(_boom).flatten_ok().collect()
        assert len(result) == 1
        assert isinstance(result[0], Err)

    def test_empty_inner_iterables_produce_no_items(self) -> None:
        def _empty(x: int) -> list[int]:
            return []

        assert Pipeline([1, 2]).map_safe(_empty).flatten_ok().collect() == []

    def test_non_iterable_ok_propagates(self) -> None:
        """flatten_ok is not error-capturing — see its docstring.

        The upstream is deliberately untyped here: ``flatten_ok`` requires
        a stream of iterables, so this misuse is a type error by design.
        """
        not_iterables: ResultPipeline[Any] = Pipeline([1]).map_safe(lambda x: x)
        with pytest.raises(TypeError):
            not_iterables.flatten_ok().collect()


# ---------------------------------------------------------------------------
# map_async_chunked_safe
# ---------------------------------------------------------------------------


class TestMapAsyncChunkedSafe:
    def test_all_succeed_wraps_in_ok(self) -> None:
        result = Pipeline([1, 2]).map_async_chunked_safe(_double).collect()
        assert result == [Ok(2), Ok(4)]

    def test_partial_failure_within_chunk(self) -> None:
        result = Pipeline([1, 2, 3]).map_async_chunked_safe(_fail_on_even).collect()
        assert result[0] == Ok(1)
        assert isinstance(result[1], Err)
        assert result[2] == Ok(3)

    def test_failure_in_one_chunk_does_not_abort_next_chunk(self) -> None:
        result = (
            Pipeline([2, 1])
            .map_async_chunked_safe(_fail_on_even, map_batch_size=1)
            .collect()
        )
        assert isinstance(result[0], Err)
        assert result[1] == Ok(1)

    def test_empty_stream(self) -> None:
        assert Pipeline([]).map_async_chunked_safe(_double).collect() == []

    def test_invalid_batch_size_raises(self) -> None:
        with pytest.raises(ValueError, match="map_batch_size must be >= 1"):
            Pipeline([1]).map_async_chunked_safe(_double, map_batch_size=0)

    def test_fatal_base_exception_is_reraised(self) -> None:
        async def _interrupt(x: int) -> int:
            raise KeyboardInterrupt()

        with pytest.raises(KeyboardInterrupt):
            Pipeline([1]).map_async_chunked_safe(_interrupt).collect()


class TestAsyncChunkedSafeThenFlattenOk:
    """The async 1-to-many stage: map_async_chunked_safe + flatten_ok."""

    def test_all_succeed_flattened_and_wrapped(self) -> None:
        result = Pipeline([1, 2]).map_async_chunked_safe(_duplicate).flatten_ok()
        assert result.collect() == [Ok(1), Ok(10), Ok(2), Ok(20)]

    def test_partial_failure_yields_single_err(self) -> None:
        async def _fail_on_two(x: int) -> list[int]:
            if x == 2:
                raise ValueError("two")
            return [x]

        result = (
            Pipeline([1, 2, 3])
            .map_async_chunked_safe(_fail_on_two)
            .flatten_ok()
            .collect()
        )
        assert result[0] == Ok(1)
        assert isinstance(result[1], Err)
        assert result[2] == Ok(3)

    def test_empty_inner_iterable(self) -> None:
        async def _nothing(x: int) -> list[int]:
            return []

        assert (
            Pipeline([1]).map_async_chunked_safe(_nothing).flatten_ok().collect() == []
        )


# ---------------------------------------------------------------------------
# ResultPipeline combinators: map_ok
# ---------------------------------------------------------------------------


def _mixed_stream() -> ResultPipeline[int]:
    return Pipeline([1, 2, 3]).map_safe(_fail_on_two_sync)


def _fail_on_two_sync(x: int) -> int:
    if x == 2:
        raise ValueError("two")
    return x


class TestMapOk:
    def test_transforms_ok_values(self) -> None:
        result = (
            Pipeline([1, 2]).map_safe(lambda x: x).map_ok(lambda x: x * 10).collect()
        )
        assert result == [Ok(10), Ok(20)]

    def test_err_passes_through_unchanged(self) -> None:
        result = _mixed_stream().map_ok(lambda x: x * 10).collect()
        assert result[0] == Ok(10)
        assert isinstance(result[1], Err)
        assert result[2] == Ok(30)

    def test_err_passes_through_at_original_position(self) -> None:
        result = _mixed_stream().map_ok(lambda x: x * 10).collect()
        assert result[0] == Ok(10)
        assert isinstance(result[1], Err)
        assert result[2] == Ok(30)

    def test_expands_via_flatten_ok(self) -> None:
        result = _mixed_stream().map_ok(lambda x: [x, x]).flatten_ok().collect()
        assert result[0] == Ok(1)
        assert result[1] == Ok(1)
        assert isinstance(result[2], Err)

    def test_exception_propagates(self) -> None:
        def _boom(x: int) -> int:
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            Pipeline([1]).map_safe(lambda x: x).map_ok(_boom).collect()

    def test_empty_stream(self) -> None:
        assert Pipeline([]).map_safe(lambda x: x).map_ok(lambda x: x).collect() == []


# ---------------------------------------------------------------------------
# ResultPipeline combinators: map_safe
# ---------------------------------------------------------------------------


class TestResultMapSafe:
    def test_ok_values_are_mapped(self) -> None:
        result = (
            Pipeline([1, 2]).map_safe(lambda x: x).map_safe(lambda x: x * 10).collect()
        )
        assert result == [Ok(10), Ok(20)]

    def test_existing_err_passes_through(self) -> None:
        result = _mixed_stream().map_safe(lambda x: x * 10).collect()
        assert result[0] == Ok(10)
        assert isinstance(result[1], Err)
        assert result[2] == Ok(30)

    def test_exception_in_func_becomes_err(self) -> None:
        result = (
            Pipeline([1])
            .map_safe(lambda x: x)
            .map_safe(_fail_on_two_sync_wrapper)
            .collect()
        )
        assert isinstance(result[0], Err)

    def test_prior_and_new_errs_accumulate(self) -> None:
        result = (
            Pipeline([1, 2, 4])
            .map_safe(_fail_on_two_sync)  # 2 -> Err
            .map_safe(_fail_on_four_sync)  # 4 -> Err
            .collect()
        )
        assert result[0] == Ok(1)
        assert isinstance(result[1], Err)
        assert isinstance(result[2], Err)


def _fail_on_two_sync_wrapper(x: int) -> int:
    raise ValueError(f"wrapped {x}")


def _fail_on_four_sync(x: int) -> int:
    if x == 4:
        raise ValueError("four")
    return x


# ---------------------------------------------------------------------------
# ResultPipeline async combinators
# ---------------------------------------------------------------------------


class TestResultMapAsyncChunkedSafe:
    def test_all_ok_values_are_mapped(self) -> None:
        result = (
            Pipeline([1, 2])
            .map_safe(lambda x: x)
            .map_async_chunked_safe(_double)
            .collect()
        )
        assert result == [Ok(2), Ok(4)]

    def test_existing_err_values_pass_through_unchanged(self) -> None:
        result = _mixed_stream().map_async_chunked_safe(_double).collect()
        assert result[0] == Ok(2)
        assert isinstance(result[1], Err)
        assert result[2] == Ok(6)

    def test_func_exception_becomes_err(self) -> None:
        result = (
            Pipeline([1, 2])
            .map_safe(lambda x: x)
            .map_async_chunked_safe(_fail_on_even)
            .collect()
        )
        assert result[0] == Ok(1)
        assert isinstance(result[1], Err)

    def test_prior_errs_and_new_errs_accumulate(self) -> None:
        result = _mixed_stream().map_async_chunked_safe(_fail_on_even).collect()
        # item 1: Ok(1); item 2: prior Err; item 3: Ok(3)
        assert result[0] == Ok(1)
        assert isinstance(result[1], Err)
        assert result[2] == Ok(3)

    def test_invalid_batch_size_raises(self) -> None:
        with pytest.raises(ValueError, match="map_batch_size must be >= 1"):
            Pipeline([1]).map_safe(lambda x: x).map_async_chunked_safe(
                _double, map_batch_size=0
            )


class TestResultAsyncChunkedSafeThenFlattenOk:
    def test_ok_values_are_expanded(self) -> None:
        result = (
            Pipeline([1, 2])
            .map_safe(lambda x: x)
            .map_async_chunked_safe(_duplicate)
            .flatten_ok()
            .collect()
        )
        assert result == [Ok(1), Ok(10), Ok(2), Ok(20)]

    def test_existing_err_passes_through(self) -> None:
        result = (
            _mixed_stream().map_async_chunked_safe(_duplicate).flatten_ok().collect()
        )
        assert result[0] == Ok(1)
        assert result[1] == Ok(10)
        assert isinstance(result[2], Err)
        assert result[3] == Ok(3)
        assert result[4] == Ok(30)


# ---------------------------------------------------------------------------
# filter_ok / on_error
# ---------------------------------------------------------------------------


class TestFilterOk:
    def test_unwraps_ok_and_drops_err(self) -> None:
        result = _mixed_stream().filter_ok().collect()
        assert result == [1, 3]

    def test_all_err_returns_empty(self) -> None:
        result = Pipeline([2, 2]).map_safe(_fail_on_two_sync).filter_ok().collect()
        assert result == []

    def test_returns_pipeline(self) -> None:
        assert isinstance(_mixed_stream().filter_ok(), Pipeline)


class TestOnError:
    def test_calls_handler_for_each_err(self) -> None:
        errors: list[Err] = []
        result = _mixed_stream().on_error(errors.append).collect()
        assert result == [1, 3]
        assert len(errors) == 1

    def test_handler_receives_err_with_exception(self) -> None:
        errors: list[Err] = []
        _mixed_stream().on_error(errors.append).collect()
        assert isinstance(errors[0].exception, ValueError)

    def test_no_errors_handler_never_called(self) -> None:
        errors: list[Err] = []
        result = Pipeline([1]).map_safe(lambda x: x).on_error(errors.append).collect()
        assert result == [1]
        assert errors == []

    def test_returns_pipeline(self) -> None:
        assert isinstance(_mixed_stream().on_error(lambda e: None), Pipeline)


# ---------------------------------------------------------------------------
# ResultPipeline.collect / direct construction
# ---------------------------------------------------------------------------


class TestResultCollect:
    def test_returns_mixed_list(self) -> None:
        result = _mixed_stream().collect()
        assert result[0] == Ok(1)
        assert isinstance(result[1], Err)
        assert result[2] == Ok(3)

    def test_direct_construction_from_result_stream(self) -> None:
        pipe: ResultPipeline[int] = ResultPipeline([Ok(1), Err(ValueError("x"))])
        assert pipe.filter_ok().collect() == [1]
