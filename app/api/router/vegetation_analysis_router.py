from __future__ import annotations

import json
import os
from tempfile import NamedTemporaryFile
from typing import Any

from fastapi import APIRouter, File, Form, UploadFile

from app.api.models import FileItem
from app.application.pipelines.vegetation_analysis_pipeline import (
    run_vegetation_analysis_pipeline,
)
from app.core.errors import ExternalServiceError, InvalidRequestError


router = APIRouter(tags=["vegetation analysis"])


def _validate_image_upload(file: UploadFile, field_name: str) -> None:
    if not file.filename:
        raise InvalidRequestError(message="上传文件缺少文件名", detail={"field": field_name})

    content_type = file.content_type or ""
    if not content_type.startswith("image/"):
        raise InvalidRequestError(
            message="仅支持图片文件",
            detail={"field": field_name, "content_type": content_type},
        )


def _validate_config(config: Any) -> dict[str, Any]:
    if not isinstance(config, dict):
        raise InvalidRequestError(message="config_json 必须是 JSON object")

    texts = config.get("texts")
    if texts is not None and not isinstance(texts, list):
        raise InvalidRequestError(message="config_json.texts 必须是数组")

    threshold = config.get("threshold")
    if threshold is not None and not isinstance(threshold, (int, float)):
        raise InvalidRequestError(message="config_json.threshold 必须是数字")

    return config


async def _build_file_item(field_name: str, file: UploadFile) -> FileItem:
    _validate_image_upload(file, field_name)
    content = await file.read()
    return FileItem(
        filename=file.filename or "",
        content_type=file.content_type or "application/octet-stream",
        data=content,
    )


def _persist_file_item(file_item: FileItem) -> FileItem:
    suffix = os.path.splitext(file_item.filename)[1] or ".jpg"
    with NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(file_item.data)
        return file_item.model_copy(update={"path": temp_file.name})


@router.post("/vegetation/analyze")
async def analyze_vegetation(
    config_json: str = Form(...),
    origin_file: UploadFile = File(...),
    ndvi_file: UploadFile = File(...),
    gndvi_file: UploadFile = File(...),
    lci_file: UploadFile = File(...),
) -> dict[str, Any]:
    upload_files: dict[str, UploadFile] = {
        "origin_file": origin_file,
        "ndvi_file": ndvi_file,
        "gndvi_file": gndvi_file,
        "lci_file": lci_file,
    }

    try:
        config_obj = json.loads(config_json)
    except json.JSONDecodeError as exc:
        raise InvalidRequestError(message="config_json 不是合法 JSON", detail=str(exc)) from exc

    config = _validate_config(config_obj)

    file_items: dict[str, FileItem] = {}
    try:
        for field_name, file in upload_files.items():
            raw_item = await _build_file_item(field_name, file)
            file_items[field_name] = _persist_file_item(raw_item)

        return await run_vegetation_analysis_pipeline(
            origin_file_item=file_items["origin_file"],
            ndvi_file_item=file_items["ndvi_file"],
            gndvi_file_item=file_items["gndvi_file"],
            lci_file_item=file_items["lci_file"],
            config=config,
        )
    except InvalidRequestError:
        raise
    except Exception as exc:
        raise ExternalServiceError(message="植被分析失败", detail=str(exc)) from exc
    finally:
        for file in upload_files.values():
            await file.close()
        for file_item in file_items.values():
            if file_item.path and os.path.exists(file_item.path):
                os.remove(file_item.path)
