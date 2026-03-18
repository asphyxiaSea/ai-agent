from __future__ import annotations

from typing import Any, Optional, cast

from pydantic import BaseModel, Field, create_model


_TYPE_MAPPING: dict[str, type[Any]] = {
    "str": str,
    "string": str,
    "int": int,
    "integer": int,
    "float": float,
    "number": float,
    "bool": bool,
    "boolean": bool,
    "list": list,
    "dict": dict,
}


def _resolve_type(type_name: str) -> type[Any]:
    mapped = _TYPE_MAPPING.get(type_name.lower())
    if mapped is None:
        raise ValueError(f"Unsupported field type: {type_name}")
    return mapped


def create_schema_model(schema_name: str, fields: list[dict[str, Any]]) -> type[BaseModel]:
    model_fields: dict[str, tuple[type[Any], Any]] = {}

    for field in fields:
        field_name = field["name"]
        field_type = _resolve_type(field.get("type", "str"))
        description = field.get("description", "")
        required = field.get("required", True)

        if required:
            annotation = field_type
            default = Field(..., description=description)
        else:
            annotation = Optional[field_type]
            default = Field(default=None, description=description)

        model_fields[field_name] = (annotation, default)

    return cast(type[BaseModel], create_model(schema_name, **cast(Any, model_fields)))


class InvoiceSchema(BaseModel):
    invoice_no: str = Field(description="Invoice number")
    seller: str = Field(description="Seller name")
    buyer: str = Field(description="Buyer name")
    total_amount: float = Field(description="Total amount")
