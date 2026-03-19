from __future__ import annotations

import json
import os
from tempfile import NamedTemporaryFile
from typing import List, Optional

from fastapi import APIRouter, File, Form, UploadFile

from app.api.models import SchemaPayload
from app.application.pipelines.pdf_structured_pipeline import (
    run_pdf_structured_pipeline,
)
from app.core.errors import ExternalServiceError, InvalidRequestError
from app.domain.build_schema import get_schema_model


router = APIRouter(tags=["files parse"])


@router.post("/files/parse")
async def parse_pdf_to_structured(
    system_prompt: Optional[str] = Form(None),
    pdf_process: Optional[str] = Form(None),
    text_process: Optional[str] = Form(None),
    schema_payload_json: str = Form(...),
    files: List[UploadFile] = File(...),
) -> dict:
    if not files:
        raise InvalidRequestError(message="No files provided")

    #  解析 schema
    try:
        schema_payload = SchemaPayload.parse_json(schema_payload_json)
    except Exception as e:
        raise InvalidRequestError(
            message="Invalid schema payload",
            detail=str(e),
        ) from e

    try:
        # 自动装配 Schema
        schema_model = get_schema_model(
            schema_name=schema_payload.schema_name,
            fields=schema_payload.fields,
        )
    except Exception as e:
        # 捕获 get_schema_model 内部可能抛出的所有错误
        raise InvalidRequestError(
            message="Schema的格式不正确，检查字段类型及格式",
            detail=str(e),
        ) from e
    

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

    uploaded_temp_paths: list[str] = []
    results: list[dict] = []
    extracted_texts: list[str] = []
    try:
        for file in files:
            if file.content_type != "application/pdf":
                raise InvalidRequestError(message="仅支持 PDF 文件", detail=file.content_type)

            content = await file.read()
            with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(content)
                temp_path = temp_file.name
                uploaded_temp_paths.append(temp_path)

            result = await run_pdf_structured_pipeline(
                pdf_path=temp_path,
                schema_model=schema_model,
                system_prompt=system_prompt or "",
                pdf_process=pdf_process_dict,
                text_process=text_process_dict,
            )
            results.append(result.get("structured_output", {}))
            extracted_texts.append(result.get("extracted_text", ""))

        return {
            "results": results,
            "extracted_texts": extracted_texts,
        }
    except InvalidRequestError:
        raise
    except Exception as exc:
        raise ExternalServiceError(message="PDF 结构化提取失败", detail=str(exc)) from exc
    finally:
        for file in files:
            await file.close()
        for temp_path in uploaded_temp_paths:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
