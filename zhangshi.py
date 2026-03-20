import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from pydantic import BaseModel
from typing import Optional
import json
import asyncio
import httpx

from app.core.errors import ExternalServiceError, InvalidRequestError


class FileItem(BaseModel):
    filename: str
    content_type: str
    data: bytes
    path: Optional[str] = None


async def sam3_segment_semantic_texts(
    *,
    file_item: FileItem,
    config: dict,
):
    """语义分割接口，返回结构：{"results": [{"name": "plant", "geojson": {...}}, ...]}"""
    is_image = file_item.content_type.startswith("image/")
    if not is_image:
        raise InvalidRequestError(
            message="不支持的文件类型",
            detail=file_item.content_type,
        )

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                "http://localhost:8002/sam3/image/segment/semantic/texts",
                files={
                    "image_file": (
                        file_item.filename,
                        file_item.data,
                        file_item.content_type,
                    )
                },
                data={
                    "config": json.dumps(config, ensure_ascii=False),
                },
            )
            resp.raise_for_status()
            return resp.json()
    except (httpx.RequestError, httpx.HTTPStatusError) as exc:
        raise ExternalServiceError(
            message="Sam3 服务异常",
            detail=str(exc),
        ) from exc


# ============ SAM3分割 + 掩膜转换 ============

def call_sam_mask(origin_img_path: str) -> dict:
    """
    用原图调用SAM3语义分割，返回 GeoJSON FeatureCollection。
    语义分割只有一个类别（plant），直接取 results[0]["geojson"]。
    """
    path = Path(origin_img_path)
    data = path.read_bytes()
    suffix = path.suffix.lower()
    content_type = "image/jpeg" if suffix in (".jpg", ".jpeg") else f"image/{suffix[1:]}"

    file_item = FileItem(
        filename=path.name,
        content_type=content_type,
        data=data,
    )
    # 语义分割 config：texts 是逗号分隔字符串，不是列表
    config = {
        "texts": ["plant"],
        "threshold": 0.25,
    }

    result = asyncio.run(sam3_segment_semantic_texts(file_item=file_item, config=config))
    return result["results"][0]["geojson"]   # 直接取第一条的 geojson


def geojson_to_mask(geojson: dict, img_shape: tuple) -> np.ndarray:
    """将 FeatureCollection 转为与光谱图同尺寸的二值掩膜"""
    h, w = img_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    for feature in geojson.get("features", []):
        geom = feature["geometry"]
        if geom["type"] == "Polygon":
            _draw_polygon(mask, geom["coordinates"])
        elif geom["type"] == "MultiPolygon":
            for poly_coords in geom["coordinates"]:
                _draw_polygon(mask, poly_coords)

    return mask


def _draw_polygon(mask: np.ndarray, coordinates: list):
    exterior = np.array(coordinates[0], dtype=np.int32)
    cv2.fillPoly(mask, [exterior], color=255)
    for hole in coordinates[1:]:
        cv2.fillPoly(mask, [np.array(hole, dtype=np.int32)], color=0)


def visualize_sam_result(origin_img_path: str, geojson: dict):
    """将SAM3分割的geojson轮廓叠加到原图上可视化"""
    img = cv2.imread(origin_img_path)
    if img is None:
        raise InvalidRequestError(
            message="原图读取失败",
            detail=origin_img_path,
        )
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    overlay = img_rgb.copy()

    # 语义分割整体只有一个类别，统一用绿色
    color = (0, 255, 0)

    for feature in geojson.get("features", []):
        geom = feature["geometry"]
        if geom["type"] == "Polygon":
            _draw_polygon_overlay(overlay, geom["coordinates"], color)
        elif geom["type"] == "MultiPolygon":
            for poly_coords in geom["coordinates"]:
                _draw_polygon_overlay(overlay, poly_coords, color)

    blended = cv2.addWeighted(img_rgb, 0.5, overlay, 0.5, 0)

    plt.figure(figsize=(14, 10))
    plt.imshow(blended)
    plt.title(f"SAM3语义分割结果  共 {len(geojson.get('features', []))} 个区域")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("sam3_result.png", dpi=150, bbox_inches="tight")
    print("SAM3分割结果已保存: sam3_result.png")
    plt.show()


def _draw_polygon_overlay(img_rgb: np.ndarray, coordinates: list, color: tuple):
    exterior = np.array(coordinates[0], dtype=np.int32)
    mask_tmp = np.zeros(img_rgb.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask_tmp, [exterior], color=255)
    for hole in coordinates[1:]:
        cv2.fillPoly(mask_tmp, [np.array(hole, dtype=np.int32)], color=0)
    img_rgb[mask_tmp > 0] = (
        img_rgb[mask_tmp > 0] * 0.4 + np.array(color) * 0.6
    ).astype(np.uint8)
    cv2.polylines(img_rgb, [exterior], isClosed=True, color=color, thickness=2)


# ============ 配置区 ============
ORIGIN_IMAGE = "images/origin.jpg"   # 用于SAM3分割的原图

IMAGE_FILES = {
    "NDVI":  "images/NDVI.jpg",
    "GNDVI": "images/GNDVI.jpg",
    "LCI":   "images/LCI.jpg",
}
# ================================


def analyze_index(img_bgr, mask, index_name):
    r_channel = img_bgr[:, :, 2]
    plant_values = r_channel[mask > 0].astype(float) / 255.0

    if len(plant_values) == 0:
        print(f"{index_name}: 未检测到植物区域")
        return None

    return {
        "index":       index_name,
        "pixel_count": len(plant_values),
        "mean":        plant_values.mean(),
        "max":         plant_values.max(),
        "min":         plant_values.min(),
        "std":         plant_values.std(),
    }


def evaluate_health(results):
    ndvi_mean  = results.get("NDVI",  {}).get("mean", 0)
    gndvi_mean = results.get("GNDVI", {}).get("mean", 0)
    lci_mean   = results.get("LCI",   {}).get("mean", 0)

    print("\n========== 植物长势评估报告 ==========")

    print(f"\n【NDVI 均值: {ndvi_mean:.3f}】整体生长活力")
    if ndvi_mean > 0.7:
        print("  ✅ 长势旺盛，植被覆盖好")
    elif ndvi_mean > 0.5:
        print("  🟡 长势一般，有改善空间")
    else:
        print("  ❌ 长势较差，建议检查光照/水分")

    print(f"\n【GNDVI 均值: {gndvi_mean:.3f}】叶绿素含量")
    if gndvi_mean > 0.6:
        print("  ✅ 叶绿素充足，氮素营养良好")
    elif gndvi_mean > 0.4:
        print("  🟡 叶绿素中等，可适量补充氮肥")
    else:
        print("  ❌ 叶绿素不足，存在缺氮风险")

    print(f"\n【LCI 均值: {lci_mean:.3f}】叶片营养状态")
    if lci_mean > 0.6:
        print("  ✅ 叶片营养充足")
    elif lci_mean > 0.4:
        print("  🟡 叶片营养中等")
    else:
        print("  ❌ 叶片营养不足")

    print("\n======================================")


def visualize(images, masks, results):
    n = len(images)
    fig, axes = plt.subplots(3, n, figsize=(5 * n, 12))

    for i, (name, img) in enumerate(images.items()):
        mask = masks[name]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        axes[0, i].imshow(img_rgb)
        axes[0, i].set_title(f"{name} 原图")
        axes[0, i].axis("off")

        axes[1, i].imshow(mask, cmap="gray")
        axes[1, i].set_title(f"{name} 植物掩膜")
        axes[1, i].axis("off")

        plant_only = img_rgb.copy()
        plant_only[mask == 0] = 0
        axes[2, i].imshow(plant_only)
        r = results.get(name)
        title = f"{name} 均值:{r['mean']:.3f}" if r else f"{name} 未检测到"
        axes[2, i].set_title(title)
        axes[2, i].axis("off")

    plt.tight_layout()
    plt.savefig("plant_analysis_result.png", dpi=150, bbox_inches="tight")
    print("\n结果图已保存: plant_analysis_result.png")
    plt.show()


# ============ 主流程 ============
images, masks, results = {}, {}, {}

print("正在调用SAM3语义分割原图...")
geojson = call_sam_mask(ORIGIN_IMAGE)
visualize_sam_result(ORIGIN_IMAGE, geojson)   # 可视化分割结果

for name, path in IMAGE_FILES.items():
    if not Path(path).exists():
        print(f"文件不存在: {path}")
        continue

    img = cv2.imread(path)
    if img is None:
        print(f"读取失败: {path}")
        continue
    mask   = geojson_to_mask(geojson, img.shape)
    result = analyze_index(img, mask, name)

    images[name] = img
    masks[name]  = mask
    if result:
        results[name] = result
        print(f"{name}: 检测到植物像素 {result['pixel_count']} 个，均值 {result['mean']:.3f}")

evaluate_health(results)
visualize(images, masks, results)