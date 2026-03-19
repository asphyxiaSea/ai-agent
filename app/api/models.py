from __future__ import annotations

from pydantic import BaseModel

class SchemaPayload(BaseModel):
    schema_name: str
    fields: list[dict]

    @classmethod
    def parse_json(cls, raw_schema: str) -> "SchemaPayload":
        cleaned = (
            raw_schema.replace("\u00a0", " ")
            .replace("\u200b", "")
            .replace("\ufeff", "")
        )
        return cls.model_validate_json(cleaned)