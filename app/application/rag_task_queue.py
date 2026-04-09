from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from time import time
from typing import Any, Literal
from uuid import uuid4

from app.application.pipelines.rag_chat_pipeline import run_rag_chat_pipeline
from app.core.errors import QueueFullError, TaskNotFoundError
from app.core.settings import (
    RAG_TASK_CLEANUP_INTERVAL_SECONDS,
    RAG_TASK_QUEUE_MAXSIZE,
    RAG_TASK_RESULT_TTL_SECONDS,
    RAG_TASK_TIMEOUT_SECONDS,
    RAG_TASK_WORKER_COUNT,
)


TaskStatus = Literal["PENDING", "RUNNING", "SUCCESS", "FAILED"]


@dataclass
class RagTaskRecord:
    task_id: str
    status: TaskStatus
    payload: dict[str, Any]
    created_at: float
    updated_at: float
    result: dict[str, Any] | None = None
    error: str | None = None


class RagTaskQueueService:
    def __init__(
        self,
        *,
        queue_maxsize: int,
        worker_count: int,
        task_timeout_seconds: float,
        result_ttl_seconds: int,
        cleanup_interval_seconds: int,
    ) -> None:
        self._queue: asyncio.Queue[tuple[str, dict[str, Any]]] = asyncio.Queue(
            maxsize=max(queue_maxsize, 1)
        )
        self._worker_count = max(worker_count, 1)
        self._task_timeout_seconds = max(task_timeout_seconds, 1.0)
        self._result_ttl_seconds = max(result_ttl_seconds, 60)
        self._cleanup_interval_seconds = max(cleanup_interval_seconds, 10)

        self._tasks: dict[str, RagTaskRecord] = {}
        self._task_lock = asyncio.Lock()
        self._workers: list[asyncio.Task[None]] = []
        self._cleanup_task: asyncio.Task[None] | None = None
        self._started = False
        self._logger = logging.getLogger(__name__)

    async def start(self) -> None:
        if self._started:
            return

        self._started = True
        self._workers = [
            asyncio.create_task(self._worker_loop(worker_index), name=f"rag-worker-{worker_index}")
            for worker_index in range(self._worker_count)
        ]
        self._cleanup_task = asyncio.create_task(self._cleanup_loop(), name="rag-cleanup")
        self._logger.info(
            "RAG task queue started: workers=%s, queue_maxsize=%s",
            self._worker_count,
            self._queue.maxsize,
        )

    async def stop(self) -> None:
        if not self._started:
            return

        self._started = False
        for worker in self._workers:
            worker.cancel()
        if self._cleanup_task:
            self._cleanup_task.cancel()

        await asyncio.gather(*self._workers, return_exceptions=True)
        if self._cleanup_task:
            await asyncio.gather(self._cleanup_task, return_exceptions=True)

        self._workers = []
        self._cleanup_task = None
        self._logger.info("RAG task queue stopped")

    async def submit_task(self, payload: dict[str, Any]) -> str:
        if not self._started:
            raise RuntimeError("RAG task queue is not started")

        if self._queue.full():
            raise QueueFullError()

        task_id = uuid4().hex
        now = time()
        record = RagTaskRecord(
            task_id=task_id,
            status="PENDING",
            payload=payload,
            created_at=now,
            updated_at=now,
        )

        async with self._task_lock:
            self._tasks[task_id] = record

        try:
            self._queue.put_nowait((task_id, payload))
        except asyncio.QueueFull as exc:
            async with self._task_lock:
                self._tasks.pop(task_id, None)
            raise QueueFullError() from exc

        return task_id

    async def get_task_snapshot(self, task_id: str) -> dict[str, Any]:
        async with self._task_lock:
            task = self._tasks.get(task_id)

        if not task:
            raise TaskNotFoundError()

        return {
            "task_id": task.task_id,
            "status": task.status,
            "created_at": self._format_timestamp(task.created_at),
            "updated_at": self._format_timestamp(task.updated_at),
            "result": task.result,
            "error": task.error,
        }

    def queue_size(self) -> int:
        return self._queue.qsize()

    async def _worker_loop(self, worker_index: int) -> None:
        while True:
            task_id, payload = await self._queue.get()
            try:
                await self._mark_running(task_id)
                started_at = time()
                result = await asyncio.wait_for(
                    run_rag_chat_pipeline(
                        question=str(payload["question"]),
                        collection_name=payload.get("collection_name"),
                        knowledge_domain=payload.get("knowledge_domain"),
                        book_id=payload.get("book_id"),
                        top_k=payload.get("top_k"),
                    ),
                    timeout=self._task_timeout_seconds,
                )
                await self._mark_success(task_id, result)
                cost_ms = int((time() - started_at) * 1000)
                self._logger.info(
                    "RAG task success: task_id=%s worker=%s cost_ms=%s queue_size=%s",
                    task_id,
                    worker_index,
                    cost_ms,
                    self._queue.qsize(),
                )
            except asyncio.TimeoutError:
                await self._mark_failed(task_id, "任务执行超时")
                self._logger.warning(
                    "RAG task timeout: task_id=%s worker=%s queue_size=%s",
                    task_id,
                    worker_index,
                    self._queue.qsize(),
                )
            except Exception as exc:  # pragma: no cover - defensive branch
                await self._mark_failed(task_id, str(exc))
                self._logger.exception(
                    "RAG task failed: task_id=%s worker=%s queue_size=%s",
                    task_id,
                    worker_index,
                    self._queue.qsize(),
                )
            finally:
                self._queue.task_done()

    async def _cleanup_loop(self) -> None:
        while True:
            await asyncio.sleep(self._cleanup_interval_seconds)
            now = time()
            async with self._task_lock:
                expired_ids = [
                    task_id
                    for task_id, task in self._tasks.items()
                    if task.status in ("SUCCESS", "FAILED")
                    and now - task.updated_at > self._result_ttl_seconds
                ]
                for task_id in expired_ids:
                    self._tasks.pop(task_id, None)
            if expired_ids:
                self._logger.info("Cleaned expired RAG tasks: count=%s", len(expired_ids))

    async def _mark_running(self, task_id: str) -> None:
        async with self._task_lock:
            task = self._tasks.get(task_id)
            if task:
                task.status = "RUNNING"
                task.updated_at = time()

    async def _mark_success(self, task_id: str, result: dict[str, Any]) -> None:
        async with self._task_lock:
            task = self._tasks.get(task_id)
            if task:
                task.status = "SUCCESS"
                task.result = result
                task.error = None
                task.updated_at = time()

    async def _mark_failed(self, task_id: str, error: str) -> None:
        async with self._task_lock:
            task = self._tasks.get(task_id)
            if task:
                task.status = "FAILED"
                task.result = None
                task.error = error
                task.updated_at = time()

    @staticmethod
    def _format_timestamp(ts: float) -> str:
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


_rag_task_queue_service = RagTaskQueueService(
    queue_maxsize=RAG_TASK_QUEUE_MAXSIZE,
    worker_count=RAG_TASK_WORKER_COUNT,
    task_timeout_seconds=RAG_TASK_TIMEOUT_SECONDS,
    result_ttl_seconds=RAG_TASK_RESULT_TTL_SECONDS,
    cleanup_interval_seconds=RAG_TASK_CLEANUP_INTERVAL_SECONDS,
)


def get_rag_task_queue_service() -> RagTaskQueueService:
    return _rag_task_queue_service


async def stop_rag_task_queue_quietly() -> None:
    with contextlib.suppress(Exception):
        await _rag_task_queue_service.stop()