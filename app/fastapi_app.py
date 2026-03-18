from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.router.pdf_structured_router import router as pdf_structured_router
from app.core.errors import AppError


def create_app() -> FastAPI:
    app = FastAPI(title="langchain app")

    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                    "detail": exc.detail,
                }
            },
        )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(pdf_structured_router, prefix="/langchain-app")
    return app


app = create_app()
