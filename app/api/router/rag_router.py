from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.application.rag_task_queue import get_rag_task_queue_service
from app.core.settings import RAG_DEFAULT_KNOWLEDGE_DOMAIN
from app.core.errors import AppError, ExternalServiceError, InvalidRequestError


router = APIRouter(tags=["rag"])


class RagChatRequest(BaseModel):
    question: str = Field(min_length=1)
    collection_name: str | None = None
    knowledge_domain: str = Field(default=RAG_DEFAULT_KNOWLEDGE_DOMAIN, min_length=1)
    book_id: str | None = None
    top_k: int | None = Field(default=None, ge=1, le=20)


@router.post("/rag/chat")
async def rag_chat(body: RagChatRequest) -> dict[str, Any]:
    try:
        rag_task_queue = get_rag_task_queue_service()
        task_id = await rag_task_queue.submit_task(
            {
                "question": body.question,
                "collection_name": body.collection_name,
                "knowledge_domain": body.knowledge_domain,
                "book_id": body.book_id,
                "top_k": body.top_k,
            }
        )
        return {
            "task_id": task_id,
            "status": "PENDING",
        }
    except InvalidRequestError:
        raise
    except AppError:
        raise
    except Exception as exc:
        raise ExternalServiceError(message="RAG 任务提交失败", detail=str(exc)) from exc


@router.get("/rag/chat/tasks/{task_id}")
async def rag_chat_task_status(task_id: str) -> dict[str, Any]:
    rag_task_queue = get_rag_task_queue_service()
    return await rag_task_queue.get_task_snapshot(task_id)


@router.get("/rag/chat/tasks/{task_id}/result")
async def rag_chat_task_result(task_id: str) -> dict[str, Any]:
    rag_task_queue = get_rag_task_queue_service()
    task = await rag_task_queue.get_task_snapshot(task_id)
    status = task["status"]

    if status in ("PENDING", "RUNNING"):
        return {
            "task_id": task_id,
            "status": status,
            "message": "任务尚未完成",
        }

    if status == "FAILED":
        return {
            "task_id": task_id,
            "status": status,
            "error": task.get("error", "任务执行失败"),
        }

    result = task.get("result") or {}
    return {
        "task_id": task_id,
        "status": status,
        "answer": result.get("answer", ""),
        "citations": result.get("citations", []),
    }
