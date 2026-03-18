from __future__ import annotations

import uvicorn

from app.fastapi_app import app


def main() -> None:
    host = "0.0.0.0"
    port = 8010
    reload = True

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload,
    )


if __name__ == "__main__":
    main()
