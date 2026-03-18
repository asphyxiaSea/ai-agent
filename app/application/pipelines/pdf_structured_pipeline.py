from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from app.workflows.pdf_structured.graph import build_pdf_structured_graph
from app.workflows.pdf_structured.state import PdfStructuredState


async def run_pdf_structured_pipeline(
    *,
    pdf_path: str,
    schema_model: type[BaseModel],
    system_prompt: str = "",
    pdf_process: dict[str, Any] | None = None,
    text_process: dict[str, Any] | None = None,
) -> dict[str, Any]:
    graph = build_pdf_structured_graph()

    state: PdfStructuredState = {
        "pdf_path": pdf_path,
        "schema_model": schema_model,
        "system_prompt": system_prompt,
    }
    if pdf_process is not None:
        state["pdf_process"] = pdf_process
    if text_process is not None:
        state["text_process"] = text_process

    result = await graph.ainvoke(state)
    return {
        "extracted_text": result.get("extracted_text", ""),
        "structured_output": result.get("structured_output", {}),
    }
