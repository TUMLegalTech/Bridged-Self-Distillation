from __future__ import annotations

import heapq
import itertools
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

ROPD_STAGE_PRIORITY: dict[Literal["teacher", "rubricator", "verifier"], int] = {
    "teacher": 0,
    "rubricator": 1,
    "verifier": 2,
}


@dataclass(frozen=True, slots=True)
class BlackOPDRequestSchedulerConfig:
    enabled: bool = True
    num_workers: int | None = None
    max_queue_size: int | None = None
    stage_priority_enabled: bool = True
    record_queue_metrics: bool = True


@dataclass(order=True, slots=True)
class ScheduledRequest:
    priority: int
    sequence_number: int
    stage: Literal["teacher", "rubricator", "verifier"] = field(compare=False)
    fn: Callable[[], Any] = field(compare=False)
    future: Future[Any] = field(compare=False)


@dataclass(frozen=True, slots=True)
class ScheduledResult:
    value: Any | None = None
    exception: BaseException | None = None


@dataclass(slots=True)
class _SchedulerMetricState:
    submitted: float = 0.0
    backpressure_hits: float = 0.0
    submit_wait_count: float = 0.0
    submit_wait_seconds: float = 0.0
    queue_depth_max: float = 0.0


class BoundedRequestScheduler:
    def __init__(
        self,
        *,
        num_workers: int,
        max_queue_size: int,
        stage_priority_enabled: bool,
        record_queue_metrics: bool = True,
        time_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        if num_workers < 1:
            raise ValueError("num_workers must be positive")
        if max_queue_size < 1:
            raise ValueError("max_queue_size must be positive")

        self.num_workers = num_workers
        self.max_queue_size = max_queue_size
        self.stage_priority_enabled = stage_priority_enabled
        self.record_queue_metrics = record_queue_metrics
        self._time_fn = time_fn
        self._condition = threading.Condition()
        self._queue: list[ScheduledRequest] = []
        self._sequence = itertools.count()
        self._metrics = _SchedulerMetricState()
        self._shutdown = False
        self._workers = [
            threading.Thread(
                target=self._worker_loop,
                name=f"ropd-request-worker-{index}",
                daemon=True,
            )
            for index in range(num_workers)
        ]
        for worker in self._workers:
            worker.start()

    def schedule(
        self,
        *,
        stage: Literal["teacher", "rubricator", "verifier"],
        fn: Callable[[], Any],
    ) -> Future[Any]:
        future: Future[Any] = Future()
        priority = ROPD_STAGE_PRIORITY[stage] if self.stage_priority_enabled else 0
        wait_started_at = self._time_fn()
        experienced_backpressure = False

        with self._condition:
            while len(self._queue) >= self.max_queue_size and not self._shutdown:
                experienced_backpressure = True
                self._condition.wait()
            if self._shutdown:
                raise RuntimeError("BoundedRequestScheduler is shut down.")

            wait_seconds = self._time_fn() - wait_started_at
            if self.record_queue_metrics:
                self._metrics.submitted += 1.0
                if experienced_backpressure:
                    self._metrics.backpressure_hits += 1.0
                    self._metrics.submit_wait_count += 1.0
                    self._metrics.submit_wait_seconds += wait_seconds

            request = ScheduledRequest(
                priority=priority,
                sequence_number=next(self._sequence),
                stage=stage,
                fn=fn,
                future=future,
            )
            heapq.heappush(self._queue, request)
            if self.record_queue_metrics:
                self._metrics.queue_depth_max = max(self._metrics.queue_depth_max, float(len(self._queue)))
            self._condition.notify_all()

        return future

    def submit(
        self,
        *,
        stage: Literal["teacher", "rubricator", "verifier"],
        fn: Callable[[], Any],
    ) -> Any:
        future = self.schedule(stage=stage, fn=fn)
        return future.result()

    def snapshot_metrics(self) -> dict[str, float]:
        with self._condition:
            submitted = self._metrics.submitted
            backpressure_hits = self._metrics.backpressure_hits
            return {
                "queue_submit_wait_seconds": self._metrics.submit_wait_seconds,
                "queue_submit_wait_count": self._metrics.submit_wait_count,
                "queue_depth_max": self._metrics.queue_depth_max,
                "queue_backpressure_rate": (backpressure_hits / submitted) if submitted > 0 else 0.0,
            }

    def shutdown(self, *, wait: bool = True) -> None:
        with self._condition:
            self._shutdown = True
            self._condition.notify_all()
        if wait:
            for worker in self._workers:
                worker.join()

    def _worker_loop(self) -> None:
        while True:
            with self._condition:
                while not self._queue and not self._shutdown:
                    self._condition.wait()
                if not self._queue and self._shutdown:
                    return
                request = heapq.heappop(self._queue)
                self._condition.notify_all()

            try:
                result = request.fn()
            except BaseException as exc:
                request.future.set_exception(exc)
                continue

            request.future.set_result(result)


__all__ = [
    "ROPD_STAGE_PRIORITY",
    "BlackOPDRequestSchedulerConfig",
    "BoundedRequestScheduler",
    "ScheduledRequest",
    "ScheduledResult",
]
