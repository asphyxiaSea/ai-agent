from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SchemaPayload(BaseModel):
    schema_name: str = Field(description="Dynamic schema name")
    fields: list[dict[str, Any]] = Field(description="Schema field definitions")
