from __future__ import annotations

import json
from typing import List, Optional

from fastapi import APIRouter, File, Form, UploadFile

from app.application.task_dispatcher import TaskType, get_task_dispatcher_service
from app.core.errors import AppError, ExternalServiceError, InvalidRequestError


router = APIRouter(tags=["files parse"])


@router.post("/files/parse")
async def parse_pdf_to_structured(
    system_prompt: Optional[str] = Form(None),
    pdf_process: Optional[str] = Form(None),
    text_process: Optional[str] = Form(None),
    schema_model_json: str = Form(...),
    files: List[UploadFile] = File(...),
) -> dict:
    if not files:
        raise InvalidRequestError(message="No files provided")

    # 解析 schema_model
    try:
        schema_model = json.loads(schema_model_json)
    except Exception as e:
        raise InvalidRequestError(
            message="Invalid schema_model JSON",
            detail=str(e),
        ) from e

    if not isinstance(schema_model, dict):
        raise InvalidRequestError(message="schema_model 必须是 JSON object")
    

    try:
        pdf_process_dict = json.loads(pdf_process) if pdf_process else None
        text_process_dict = json.loads(text_process) if text_process else None
    except json.JSONDecodeError as exc:
        raise InvalidRequestError(
            message="Invalid pdf_process or text_process JSON",
            detail=str(exc),
        ) from exc

    if pdf_process_dict is not None and not isinstance(pdf_process_dict, dict):
        raise InvalidRequestError(message="pdf_process 必须是 JSON object")
    if text_process_dict is not None and not isinstance(text_process_dict, dict):
        raise InvalidRequestError(message="text_process 必须是 JSON object")

    try:
        file_payloads: list[dict] = []
        for file in files:
            if file.content_type != "application/pdf":
                raise InvalidRequestError(message="仅支持 PDF 文件", detail=file.content_type)

            content = await file.read()
            file_payloads.append(
                {
                    "filename": file.filename,
                    "content_type": file.content_type,
                    "data": content,
                }
            )

        dispatcher = get_task_dispatcher_service()
        task_id = await dispatcher.submit_task(
            task_type=TaskType.PDF_STRUCTURED,
            payload={
                "schema_model": schema_model,
                "system_prompt": system_prompt or "",
                "pdf_process": pdf_process_dict,
                "text_process": text_process_dict,
                "files": file_payloads,
            },
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
        raise ExternalServiceError(message="PDF 任务提交失败", detail=str(exc)) from exc
    finally:
        for file in files:
            await file.close()


@router.get("/files/parse/tasks/{task_id}")
async def parse_pdf_task_status(task_id: str) -> dict:
    dispatcher = get_task_dispatcher_service()
    task = await dispatcher.get_task_snapshot(task_id)
    if task.get("task_type") != TaskType.PDF_STRUCTURED:
        raise InvalidRequestError(message="任务类型不匹配")
    return task


@router.get("/files/parse/tasks/{task_id}/result")
async def parse_pdf_task_result(task_id: str) -> dict:
    dispatcher = get_task_dispatcher_service()
    task = await dispatcher.get_task_snapshot(task_id)
    if task.get("task_type") != TaskType.PDF_STRUCTURED:
        raise InvalidRequestError(message="任务类型不匹配")

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
        "results": result.get("results", []),
        "extracted_texts": result.get("extracted_texts", []),
    }
