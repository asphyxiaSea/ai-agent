import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path













# ============ 配置区 ============
IMAGE_FILES = {
    "NDVI": "images/NDVI.jpg",
    "GNDVI": "images/GNDVI.jpg",
    "LCI": "images/LCI.jpg",
}
# ================================

def extract_vegetation_mask(img_bgr):
    """
    从伪彩色图中提取植物掩膜
    植物区域 = 橙红色（高值区域）
    背景区域 = 蓝绿色（低值区域）
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    
    # 橙红色范围（对应指数高值区域）
    mask_orange = cv2.inRange(hsv, 
        np.array([5, 100, 100]),    # 橙色下限
        np.array([25, 255, 255])    # 橙色上限
    )
    mask_red = cv2.inRange(hsv,
        np.array([0, 100, 100]),
        np.array([5, 255, 255])
    )
    mask = cv2.bitwise_or(mask_orange, mask_red)
    
    # 形态学处理，去除噪点
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # 去小噪点
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # 填小空洞
    
    return mask

def analyze_index(img_bgr, mask, index_name):
    """
    用R通道近似还原指数相对值（伪彩色图的局限）
    橙红色 R通道值越高 → 指数值越高
    """
    r_channel = img_bgr[:, :, 2]  # OpenCV是BGR，2是R
    
    plant_values = r_channel[mask > 0].astype(float) / 255.0
    
    if len(plant_values) == 0:
        print(f"{index_name}: 未检测到植物区域，请调整掩膜阈值")
        return None
    
    result = {
        "index": index_name,
        "pixel_count": len(plant_values),
        "mean": plant_values.mean(),
        "max": plant_values.max(),
        "min": plant_values.min(),
        "std": plant_values.std(),
    }
    return result

def evaluate_health(results):
    """根据三个指数综合评估长势"""
    ndvi_mean  = results.get("NDVI",  {}).get("mean", 0)
    gndvi_mean = results.get("GNDVI", {}).get("mean", 0)
    lci_mean   = results.get("LCI",   {}).get("mean", 0)
    
    print("\n========== 植物长势评估报告 ==========")
    
    # NDVI：整体活力
    print(f"\n【NDVI 均值: {ndvi_mean:.3f}】整体生长活力")
    if ndvi_mean > 0.7:
        print("  ✅ 长势旺盛，植被覆盖好")
    elif ndvi_mean > 0.5:
        print("  🟡 长势一般，有改善空间")
    else:
        print("  ❌ 长势较差，建议检查光照/水分")
    
    # GNDVI：叶绿素/氮素
    print(f"\n【GNDVI 均值: {gndvi_mean:.3f}】叶绿素含量")
    if gndvi_mean > 0.6:
        print("  ✅ 叶绿素充足，氮素营养良好")
    elif gndvi_mean > 0.4:
        print("  🟡 叶绿素中等，可适量补充氮肥")
    else:
        print("  ❌ 叶绿素不足，存在缺氮风险")
    
    # LCI：叶片营养
    print(f"\n【LCI 均值: {lci_mean:.3f}】叶片营养状态")
    if lci_mean > 0.6:
        print("  ✅ 叶片营养充足")
    elif lci_mean > 0.4:
        print("  🟡 叶片营养中等")
    else:
        print("  ❌ 叶片营养不足")
    
    print("\n======================================")

def visualize(images, masks, results):
    """可视化原图、掩膜、植物区域"""
    n = len(images)
    fig, axes = plt.subplots(3, n, figsize=(5 * n, 12))
    
    for i, (name, img) in enumerate(images.items()):
        mask = masks[name]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 第一行：原图
        axes[0, i].imshow(img_rgb)
        axes[0, i].set_title(f"{name} 原图")
        axes[0, i].axis("off")
        
        # 第二行：掩膜
        axes[1, i].imshow(mask, cmap="gray")
        axes[1, i].set_title(f"{name} 植物掩膜")
        axes[1, i].axis("off")
        
        # 第三行：提取结果
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

for name, path in IMAGE_FILES.items():
    if not Path(path).exists():
        print(f"文件不存在: {path}")
        continue
    
    img = cv2.imread(path)
    mask = extract_vegetation_mask(img)
    result = analyze_index(img, mask, name)
    
    images[name] = img
    masks[name] = mask
    if result:
        results[name] = result
        print(f"{name}: 检测到植物像素 {result['pixel_count']} 个，均值 {result['mean']:.3f}")

evaluate_health(results)
visualize(images, masks, results)
