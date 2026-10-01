"""
Response formatter: converts raw query engine output into the final
QueryResponse shape with a polished natural-language answer.
"""
from __future__ import annotations
from app.models.schemas import Intent, QueryResponse, StructuredQuery


class ResponseFormatter:
    """Wraps query engine results into a consistent QueryResponse."""

    def format(
        self,
        question: str,
        query: StructuredQuery,
        engine_result: dict,
    ) -> QueryResponse:
        return QueryResponse(
            question=question,
            intent=query.intent.value,
            answer=engine_result.get("answer", "No answer available."),
            data=engine_result.get("data", {}),
            structured_query=query.model_dump(mode="json"),
        )

    def format_error(
        self,
        question: str,
        error_message: str,
        intent: str = "ERROR",
    ) -> QueryResponse:
        return QueryResponse(
            question=question,
            intent=intent,
            answer=error_message,
            data={},
            structured_query=None,
        )
