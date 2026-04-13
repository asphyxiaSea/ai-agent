from __future__ import annotations

from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from app.core.settings import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_PROVIDER,
    DEFAULT_TEMPERATURE,
    OLLAMA_BASE_URL,
    RAG_CHROMA_COLLECTION,
    RAG_RETRIEVAL_TOP_K,
)
from app.infra.clients.chroma_client import (
    build_citations,
    search_chroma,
)


def _retrieve_context(
    *,
    question: str,
    collection_name: str | None = None,
    knowledge_domain: str | None = None,
    book_id: str | None = None,
    top_k: int | None = None,
) -> dict[str, Any]:
    normalized_question = question.strip()
    selected_collection = collection_name or RAG_CHROMA_COLLECTION
    selected_top_k = top_k or RAG_RETRIEVAL_TOP_K
    selected_knowledge_domain = (knowledge_domain or "").strip()
    selected_book_id = (book_id or "").strip()

    metadata_filter: dict[str, Any] | None = None
    filter_map: dict[str, Any] = {}
    if selected_knowledge_domain:
        filter_map["domain"] = selected_knowledge_domain
    if selected_book_id:
        filter_map["book_id"] = selected_book_id
    if filter_map:
        metadata_filter = filter_map

    docs_with_scores = search_chroma(
        query=normalized_question,
        top_k=selected_top_k,
        collection_name=selected_collection,
        metadata_filter=metadata_filter,
    )
    return {
        "docs_with_scores": docs_with_scores,
        "citations": build_citations(docs_with_scores),
    }


def _generate_answer(
    *,
    question: str,
    docs_with_scores: list[tuple[Document, float]],
) -> str:
    if not docs_with_scores:
        return "没有检索到相关资料，当前无法基于知识库给出可靠答案。"

    contexts = []
    for idx, (doc, score) in enumerate(docs_with_scores, start=1):
        source = doc.metadata.get("source", "unknown")
        chunk_idx = doc.metadata.get("chunk_index", -1)
        contexts.append(
            f"[{idx}] source={source}, chunk={chunk_idx}, score={score:.4f}\n{doc.page_content}"
        )

    system_prompt = (
        "你是企业知识库问答助手。"
        "只能根据给定上下文回答，禁止编造。"
        "若证据不足，请明确说明“依据不足”。"
        "答案请精炼，并在末尾给出引用编号，如 [1][2]。"
    )
    human_prompt = f"用户问题：{question}\n\n" "可用上下文如下：\n" + "\n\n".join(contexts)

    model = init_chat_model(
        DEFAULT_MODEL_NAME,
        model_provider=DEFAULT_MODEL_PROVIDER,
        base_url=OLLAMA_BASE_URL,
        temperature=DEFAULT_TEMPERATURE,
        max_tokens=DEFAULT_MAX_TOKENS,
    )
    result = model.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ]
    )
    return str(result.content)


async def run_rag_chat_pipeline(
    *,
    question: str,
    collection_name: str | None = None,
    knowledge_domain: str | None = None,
    book_id: str | None = None,
    top_k: int | None = None,
) -> dict[str, Any]:
    retrieval_result = _retrieve_context(
        question=question,
        collection_name=collection_name,
        knowledge_domain=knowledge_domain,
        book_id=book_id,
        top_k=top_k,
    )
    docs_with_scores = retrieval_result.get("docs_with_scores", [])
    answer = _generate_answer(
        question=question,
        docs_with_scores=docs_with_scores,
    )
    return {
        "answer": answer,
        "citations": retrieval_result.get("citations", []),
    }
