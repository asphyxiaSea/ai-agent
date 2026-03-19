
from pydantic import BaseModel, create_model
from typing import Optional, Literal, List
import json
import hashlib

# 结构字段（参与 identity）
STRUCT_KEYS = {"name", "type", "required", "items", "enum"}

# 元数据字段（不参与 identity）
META_KEYS = {"description", "title", "example", "ui", "prompt"}

# 结构 Canonical 化（决定缓存）
def canonicalize_struct_field(f: dict) -> dict:
    return {
        "name": f["name"],
        "type": f["type"],
        "required": bool(f.get("required", False)),
        "items": f.get("items"),
        "enum": tuple(f["enum"]) if "enum" in f else None,
    }


def canonicalize_struct_fields(fields: list[dict]) -> list[dict]:
    return sorted(
        (canonicalize_struct_field(f) for f in fields),
        key=lambda x: x["name"],
    )


# 结构 Identity（唯一 Model Key）
def struct_identity(schema_name: str, fields: list[dict]) -> str:
    payload = {
        "schema_name": schema_name,
        "fields": canonicalize_struct_fields(fields),
    }
    raw = json.dumps(payload, sort_keys=True)
    return hashlib.md5(raw.encode()).hexdigest()


# 结构模型缓存（强缓存）
_STRUCT_MODEL_CACHE: dict[str, type[BaseModel]] = {}


# 元数据热更新机制（核心）
def extract_metadata(f: dict) -> dict:
    return {k: v for k, v in f.items() if k not in STRUCT_KEYS}


# 构建字段，但允许 metadata 后注入
def build_struct_model(
    schema_name: str,
    fields: list[dict],
) -> type[BaseModel]:

    sid = struct_identity(schema_name, fields)
    model_name = f"{schema_name}_{sid[:8]}"

    if sid in _STRUCT_MODEL_CACHE:
        return _STRUCT_MODEL_CACHE[sid]

    model_fields = {}

    for f in fields:
        py_type = resolve_type(f)

        default = ... if f.get("required", False) else None
        model_fields[f["name"]] = (Optional[py_type], default)

    model = create_model(model_name, **model_fields)

    _STRUCT_MODEL_CACHE[sid] = model
    return model


# Field-level 元数据更新
def apply_field_metadata(model: type[BaseModel], fields: list[dict]):
    for f in fields:
        meta = extract_metadata(f)
        if not meta:
            continue

        # model.model_fields[name] 是 ModelField
        field = model.model_fields.get(f["name"])
        if not field:
            continue

        # 更新 description
        if "description" in meta:
            field.description = meta["description"]

        # 更新 json_schema_extra（如果是 None 先创建字典）
        extras = field.json_schema_extra or {}
        extras.update({k: v for k, v in meta.items() if k != "description"})
        field.json_schema_extra = extras


# Schema-level 元数据
def apply_schema_metadata(
    model: type[BaseModel],
    schema_meta: dict,
):
    model.model_config["title"] = schema_meta.get("title")
    model.model_config["json_schema_extra"] = schema_meta


# 统一入口（你对外只暴露这个）
def get_schema_model(
    schema_name: str,
    fields: list[dict],
    *,
    schema_meta: dict | None = None,
) -> type[BaseModel]:

    model = build_struct_model(schema_name, fields)

    apply_field_metadata(model, fields)

    if schema_meta:
        apply_schema_metadata(model, schema_meta)

    return model


TYPE_MAP = {
    "string": str,
    "float": float,
    "integer": int,
    "boolean": bool,
}


def resolve_type(f: dict) -> type:
    t = f.get("type")
    if not t:
        raise ValueError("Field missing 'type' key")

    # enum → Literal
    if t == "enum":
        values = f.get("enum")
        if not values or not isinstance(values, list):
            raise ValueError("enum field must define a non-empty enum list")
        first_type = type(values[0])
        if not all(isinstance(v, first_type) for v in values):
            raise ValueError("All enum values must be of the same type")
        return Literal[tuple(values)]  # type: ignore

    # array → List[T]
    if t == "array":
        item_type_name = f.get("items", "string")
        item_type = TYPE_MAP.get(item_type_name)
        if not item_type:
            raise ValueError(f"Unsupported array item type: {item_type_name}")
        return List[item_type]  # type: ignore

    # primitive
    py_type = TYPE_MAP.get(t)
    if not py_type:
        raise ValueError(f"Unsupported field type: {t}")
    return py_type
