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
from app.domain.schemas import create_schema_model

router = APIRouter(tags=["langchain-app"])


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

    try:
        payload = SchemaPayload.model_validate(json.loads(schema_payload_json))
    except Exception as exc:
        raise InvalidRequestError(message="Invalid schema payload", detail=str(exc)) from exc

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

    schema_model = create_schema_model(
        schema_name=payload.schema_name,
        fields=payload.fields,
    )

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
