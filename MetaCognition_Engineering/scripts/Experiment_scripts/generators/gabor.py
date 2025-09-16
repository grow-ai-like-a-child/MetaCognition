# generators/gabor.py
# -*- coding: utf-8 -*-
"""
Gabor task · 新难度梯度版（按距 45° 的角度差定义难度）
- 难度层 gabor_level ∈ {1..7} 映射为与 45° 的角度差 d（度）：
    1:  2.0   （最难，最靠近判别边界）
    2:  5.0
    3: 10.0
    4: 15.0
    5: 22.5
    6: 30.0
    7: 40.0  （最易）
  生成时再随机（可复现）决定在 45° 两侧（+/-）以及象限（+k·90°）。

- 统一入口：generate(params, out_dir) -> str (保存路径)
- 兼容老参数：可直接给 theta_deg（或 delta_deg/delta_theta 视为绝对朝向），以及 freq/spatial_freq、position/location_id 等。
- 提供 GT：gabor_ground_truth(theta_deg) -> "vertical"/"horizontal"

依赖：numpy、matplotlib
"""

import os
import hashlib
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 无显示环境安全
import matplotlib.pyplot as plt

# ========== 难度层映射 ==========
GABOR_LEVEL_TO_D = {
    1: 2.0,
    2: 5.0,
    3: 10.0,
    4: 15.0,
    5: 22.5,
    6: 30.0,
    7: 40.0,
}

# ========== 枚举 ==========
COLOR_LIST = ["gray","blue","red","green","yellow","purple","cyan","orange"]
POSITION_LIST = ["center","left","right","top","bottom","random"]  # 6类

# ========== 核心渲染 ==========
def _make_gabor(size, cycles_per_image, theta_deg, sigma_ratio=0.22, phase=0.0):
    """生成 [0,1] 的 Gabor patch（高斯窗×正弦条纹）"""
    s = size
    x = np.linspace(-0.5, 0.5, s, endpoint=False)
    X, Y = np.meshgrid(x, x)
    th = np.deg2rad(theta_deg)
    Xr = X * np.cos(th) + Y * np.sin(th)

    grating = np.cos(2 * np.pi * cycles_per_image * Xr + phase)
    gauss = np.exp(-(X**2 + Y**2) / (2 * (sigma_ratio**2)))
    gabor = grating * gauss

    # 归一化到 [0,1]
    gabor = (gabor - gabor.min()) / (gabor.max() - gabor.min() + 1e-8)
    return gabor

def _rgb_for(color: str):
    """整图颜色滤镜（R,G,B）"""
    return {
        "gray":   (1.0, 1.0, 1.0),
        "blue":   (0.3, 0.3, 1.0),
        "red":    (1.0, 0.3, 0.3),
        "green":  (0.3, 1.0, 0.3),
        "yellow": (1.0, 1.0, 0.3),
        "purple": (0.7, 0.3, 0.9),
        "cyan":   (0.3, 1.0, 1.0),
        "orange": (1.0, 0.6, 0.2),
    }.get(color, (1.0, 1.0, 1.0))

def _composite_on_canvas(image_size, patch_size, position, patch_gray, bg_gray=0.5):
    """把 Gabor patch 融合到画布上，返回 RGB ndarray"""
    canvas_gray = np.ones((image_size, image_size), dtype=float) * bg_gray

    # 放置位置
    margin = (image_size - patch_size) // 8
    if position == "center":
        x = (image_size - patch_size) // 2; y = (image_size - patch_size) // 2
    elif position == "left":
        x, y = margin, (image_size - patch_size) // 2
    elif position == "right":
        x, y = image_size - patch_size - margin, (image_size - patch_size) // 2
    elif position == "top":
        x, y = (image_size - patch_size) // 2, image_size - patch_size - margin
    elif position == "bottom":
        x, y = (image_size - patch_size) // 2, margin
    elif position == "random":
        xy = np.random.randint(0, image_size - patch_size + 1, size=2)
        x, y = int(xy[0]), int(xy[1])
    else:
        raise ValueError(f"Unknown position: {position}")

    # 高斯窗 alpha 融合
    s = patch_size
    xv = np.linspace(-0.5, 0.5, s, endpoint=False)
    X, Y = np.meshgrid(xv, xv)
    sigma = 0.22
    env = np.exp(-(X**2 + Y**2) / (2 * (sigma**2)))
    env = (env - env.min()) / (env.max() - env.min() + 1e-8)

    crop = canvas_gray[y:y+s, x:x+s]
    canvas_gray[y:y+s, x:x+s] = env * patch_gray + (1 - env) * crop
    return canvas_gray

def _save_rgb(rgb_img, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig = plt.figure(figsize=(6, 6))
    plt.imshow(rgb_img); plt.axis('off')
    plt.subplots_adjust(0, 0, 1, 1)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

# ========== GT ==========
def gabor_ground_truth(theta_deg: float) -> str:
    """
    基于与最近 90°倍数的夹角，<45° 判为 vertical，>=45° 判为 horizontal。
    """
    a = abs(theta_deg) % 90.0
    a = min(a, 90.0 - a)
    return "vertical" if a < 45.0 else "horizontal"

# ========== 参数规范化 ==========
def _coerce_params(params: dict) -> dict:
    """
    入参支持（按优先级）：
      - gabor_level: 1..7    # 新难度层（推荐）
      - gabor_distance_deg: float  # 与 45° 的角度差 d，直接指定
      - theta_deg or delta_deg/delta_theta  # 绝对朝向（0..180），兼容旧字段
      - freq / spatial_freq: 空间频率（每个 patch 的周期数），默认 8
      - color / color_id: 颜色滤镜
      - position / location_id: "center|left|right|top|bottom|random"
      - img_size (默认 256), patch_size (默认 128), bg_gray (默认 0.5)
      - seed: 若需可复现的随机“侧/象限”选择，建议提供；不提供时按参数派生一个稳定 seed
    """
    # 频率
    freq = params.get("freq", params.get("spatial_freq", 8))
    freq = float(freq)

    # 颜色
    if "color" in params:
        color = params["color"]
    elif "color_id" in params:
        cid = int(params["color_id"]) % len(COLOR_LIST)
        color = COLOR_LIST[cid]
    else:
        color = "gray"

    # 位置
    if "position" in params:
        position = params["position"]
        if position not in POSITION_LIST:
            raise ValueError(f"Unknown position: {position}")
    elif "location_id" in params:
        lid = int(params["location_id"]) % len(POSITION_LIST)
        position = POSITION_LIST[lid]
    else:
        position = "center"

    img_size  = int(params.get("img_size", params.get("image_size", 256)))
    patch_size = int(params.get("patch_size", 128))
    bg_gray = float(params.get("bg_gray", 0.5))

    # 角度来源：优先难度层/距离 → 绝对朝向；否则使用显式 theta
    theta_explicit = params.get("theta_deg", params.get("delta_deg", params.get("delta_theta", None)))
    level = params.get("gabor_level", None)
    dist  = params.get("gabor_distance_deg", None)

    # 确定 seed（用于选择 ± 与象限）
    seed = params.get("seed", None)
    if seed is None:
        base = f"{level}|{dist}|{theta_explicit}|{freq}|{color}|{position}|{img_size}|{patch_size}|{bg_gray}"
        seed = (abs(hash(base)) ^ 0x13579BDF) & 0xFFFFFFFF
    seed = int(seed)

    # 计算 theta
    if level is not None or dist is not None:
        # from difficulty
        if level is not None:
            level = int(level)
            if level not in GABOR_LEVEL_TO_D:
                raise ValueError("gabor_level must be 1..7")
            d = float(GABOR_LEVEL_TO_D[level])
        else:
            d = float(dist)
            if not (0.0 <= d <= 45.0):
                raise ValueError("gabor_distance_deg must be within [0,45]")

        # 用 seed 确定两侧与象限
        rng = np.random.RandomState(seed)
        side = rng.choice([-1.0, +1.0])      # 在 45° 两侧
        quad_k = int(rng.choice([0,1,2,3]))  # 象限 0..3
        theta_deg = (45.0 + side * d + 90.0 * quad_k) % 180.0
        gabor_level = level if level is not None else None
        gabor_distance_deg = d
    else:
        if theta_explicit is None:
            raise ValueError("Provide gabor_level / gabor_distance_deg OR explicit theta (theta_deg/delta_deg/delta_theta).")
        theta_deg = float(theta_explicit) % 180.0
        # 若是显式角度，没有难度元数据
        gabor_level = None
        # 可选：记录与 45° 的距离
        a = abs(theta_deg) % 90.0
        a = min(a, 90.0 - a)
        gabor_distance_deg = abs(45.0 - a)

    return {
        "theta_deg": theta_deg,
        "freq": freq,
        "color": color,
        "position": position,
        "img_size": img_size,
        "patch_size": patch_size,
        "bg_gray": bg_gray,
        "seed": seed,
        "gabor_level": gabor_level,
        "gabor_distance_deg": gabor_distance_deg
    }

# ========== 命名 ==========
def _stable_filename(p: dict) -> str:
    """
    新模式（有 gabor_level）：GAB_L{L}_theta{deg}_f{freq}_{pos}_{color}_{seed}.png
    旧模式（无 level）：     GAB_theta{deg}_f{freq}_{pos}_{color}_{seed}.png
    """
    theta_int = int(round(p["theta_deg"]))
    freq_int  = int(round(p["freq"]))
    seed_hex  = f"{p['seed']:08x}"
    if p["gabor_level"] is not None:
        return f"GAB_L{p['gabor_level']}_theta{theta_int}_f{freq_int}_{p['position']}_{p['color']}_{seed_hex}.png"
    return f"GAB_theta{theta_int}_f{freq_int}_{p['position']}_{p['color']}_{seed_hex}.png"

# ========== 统一入口 ==========
def generate(params: dict, out_dir: str) -> str:
    """
    根据 params 生成单张 Gabor PNG 并返回路径。
    示例（新难度）：
      {
        "gabor_level": 3,      # 1..7
        "freq": 8,
        "color_id": 2,
        "location_id": 0,
        "img_size": 256,
        "patch_size": 128,
        "seed": 42
      }
    示例（老参数）：
      {
        "theta_deg": 55,       # 或 delta_deg/delta_theta
        "freq": 10,
        "color": "blue",
        "position": "left"
      }
    """
    p = _coerce_params(params)
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, _stable_filename(p))

    # 缓存命中直接返回
    if os.path.exists(save_path):
        return save_path

    # 渲染
    if p["seed"] is not None:
        np.random.seed(p["seed"])
    g = _make_gabor(p["patch_size"], p["freq"], p["theta_deg"])
    canvas_gray = _composite_on_canvas(p["img_size"], p["patch_size"], p["position"], g, bg_gray=p["bg_gray"])
    rgb = np.array(_rgb_for(p["color"])).reshape(1,1,3)
    canvas_rgb = np.clip(canvas_gray[..., None] * rgb, 0, 1)
    _save_rgb(canvas_rgb, save_path)
    return save_path

# ========== 便捷：返回 GT（vertical/horizontal） ==========
def gabor_gt_from_params(params: dict) -> str:
    """
    方便外部在只有 params 的情况下拿到 GT。
    """
    p = _coerce_params(params)
    return gabor_ground_truth(p["theta_deg"])

# 可选：保留 CLI 或批量功能（此处留空，工程按题号即时生成）
if __name__ == "__main__":
    pass
