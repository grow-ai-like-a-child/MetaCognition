#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Grid（XO 网格）task · 新难度梯度版（修复：形状中英文别名统一）
- 支持 shape 输入为中文（方形/圆形/三角形/菱形）或英文（Square/Circle/Triangle/Diamond，大小写均可），
  内部统一规范为中文以供渲染；文件名仍输出英文 slug，便于跨平台与检索。
- 统一入口：generate(params, out_dir) -> image_path
- GT：xo_ground_truth(params) -> {"more_symbol": "symA|symB|equal", "symA": "...", "symB": "..."}

依赖：Pillow
"""

import os
import csv
import math
import random
import argparse
from pathlib import Path
from typing import List, Tuple, Dict

from PIL import Image, ImageDraw, ImageFont

# =========================
# 1) 维度定义
# =========================

# 新难度梯度（symA 的目标百分比；多数方再随机决定到 A 或 B）
GRID_LEVEL_TO_PCT = {
    1: 52,  # 52/48
    2: 60,
    3: 65,
    4: 70,
    5: 75,
    6: 80,
    7: 90,  # 90/10
}

# 兼容旧 ratio（不建议新数据使用）
PERCENTAGE_LEVELS = ["100%O", "90%O", "70%O", "50/50", "70%X", "90%X", "100%X"]

# —— 统一形状别名（中英文都支持）——
SHAPES_CANON = ["方形", "圆形", "三角形", "菱形"]  # 内部规范形状（中文）
SHAPE_EN_SLUG = {"方形": "Square", "圆形": "Circle", "三角形": "Triangle", "菱形": "Diamond"}
SHAPE_ALIASES = {
    # 中文本体
    "方形": "方形", "圆形": "圆形", "三角形": "三角形", "菱形": "菱形",
    # 英文（含大小写）
    "Square": "方形", "Circle": "圆形", "Triangle": "三角形", "Diamond": "菱形",
    "square": "方形", "circle": "圆形", "triangle": "三角形", "diamond": "菱形",
}

def normalize_shape_token(s: str) -> str:
    return SHAPE_ALIASES.get(s, s)

SYMBOL_PAIRS = {
    "XO": ("O", "X"),
    "AB": ("A", "B"),
    "SD": ("S", "D"),
    "ZP": ("Z", "P"),
}

BASIC_COLORS = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "red": (220, 20, 60),
    "blue": (30, 144, 255),
    "green": (34, 139, 34),
    "orange": (255, 140, 0),
    "cyan": (0, 180, 180),
    "pink": (255, 105, 180),
    "brown": (139, 69, 19),
    "yellow": (255, 215, 0),
    "navy": (0, 0, 128),
    "lime": (173, 255, 47),
    "magenta": (255, 0, 255),
    "gray": (128, 128, 128),
}

# 8 种 symbol 颜色，背景统一为灰色
SYMBOL_COLORS = [
    "red",
    "blue", 
    "green",
    "orange",
    "cyan",
    "pink",
    "yellow",
    "brown",
]

# 背景颜色统一为灰色
BACKGROUND_COLOR = "gray"


def parse_color_pair(name: str) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    # 新的颜色配置：symbol 颜色 + 灰色背景
    if name in SYMBOL_COLORS:
        return BASIC_COLORS[name], BASIC_COLORS[BACKGROUND_COLOR]
    # 兼容旧格式 "a_b"
    elif "_" in name:
        a, b = name.split("_")
        return BASIC_COLORS[a], BASIC_COLORS[b]
    else:
        # 默认使用红色 symbol + 灰色背景
        return BASIC_COLORS["red"], BASIC_COLORS[BACKGROUND_COLOR]


# =========================
# 2) 字体
# =========================

def load_font(px: int, font_path: str = "") -> ImageFont.FreeTypeFont:
    if font_path and os.path.exists(font_path):
        try:
            return ImageFont.truetype(font_path, px)
        except Exception:
            pass
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/arialuni.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, px)
            except Exception:
                continue
    return ImageFont.load_default()


# =========================
# 3) 掩码（形状→网格坐标）
# =========================

def mask_square(N: int) -> List[Tuple[int, int]]:
    return [(i, j) for i in range(N) for j in range(N)]


def mask_circle(N: int, margin: float = 0.5) -> List[Tuple[int, int]]:
    cx = (N - 1) / 2.0
    cy = (N - 1) / 2.0
    r = (N / 2.0) - margin
    S = []
    for i in range(N):
        for j in range(N):
            if (i - cy) ** 2 + (j - cx) ** 2 <= r ** 2 + 1e-9:
                S.append((i, j))
    return S


def point_in_triangle(px, py, ax, ay, bx, by, cx, cy) -> bool:
    # barycentric
    v0x, v0y = cx - ax, cy - ay
    v1x, v1y = bx - ax, by - ay
    v2x, v2y = px - ax, py - ay
    dot00 = v0x * v0x + v0y * v0y
    dot01 = v0x * v1x + v0y * v1y
    dot02 = v0x * v2x + v0y * v2y
    dot11 = v1x * v1x + v1y * v1y
    dot12 = v1x * v2x + v1y * v2y
    denom = dot00 * dot11 - dot01 * dot01
    if abs(denom) < 1e-9:
        return False
    inv = 1.0 / denom
    u = (dot11 * dot02 - dot01 * dot12) * inv
    v = (dot00 * dot12 - dot01 * dot02) * inv
    return (u >= -1e-9) and (v >= -1e-9) and (u + v <= 1.0 + 1e-9)


def mask_triangle_equilateral(N: int, margin_cells: int = 1) -> List[Tuple[int, int]]:
    # 等边三角形（底朝下）
    top = ((N - 1) / 2.0, margin_cells)
    left = (margin_cells, N - 1 - margin_cells)
    right = (N - 1 - margin_cells, N - 1 - margin_cells)
    S = []
    for i in range(N):
        for j in range(N):
            if point_in_triangle(j + 0.5, i + 0.5, *top, *left, *right):
                S.append((i, j))
    return S


def mask_diamond(N: int) -> List[Tuple[int, int]]:
    cx = (N - 1) / 2.0
    cy = (N - 1) / 2.0
    r = int((N - 1) / 2)
    S = []
    for i in range(N):
        for j in range(N):
            if abs(i - cy) + abs(j - cx) <= r + 1e-9:
                S.append((i, j))
    return S


def get_mask_coords(shape: str, N: int) -> List[Tuple[int, int]]:
    shape = normalize_shape_token(shape)
    if shape == "方形":
        return mask_square(N)
    elif shape == "圆形":
        return mask_circle(N)
    elif shape == "三角形":
        return mask_triangle_equilateral(N)
    elif shape == "菱形":
        return mask_diamond(N)
    else:
        raise ValueError(f"未知形状: {shape}")


# =========================
# 4) 绘制
# =========================

def draw_text_centered(draw: ImageDraw.ImageDraw, xy: Tuple[int, int], text: str,
                       font: ImageFont.ImageFont, fill=(0, 0, 0)):
    x, y = xy
    bbox = draw.textbbox((0, 0), text, font=font, anchor=None)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    draw.text((x - tw / 2, y - th / 2), text, font=font, fill=fill)


def render_grid_image(
    grid_size: int,
    shape: str,
    pctA: int,
    symbol_key: str,
    color_key: str,
    filter_mode: str,
    canvas_px: int,
    seed: int,
    font_k: float = 11.0,
    font_path: str = ""
) -> Tuple[Image.Image, Dict]:
    """
    pctA: symA 的百分比（多数方已经包含在 pctA 内部）
    """
    random.seed(seed)

    # 颜色
    fg_color, bg_color = parse_color_pair(color_key)

    # 画布
    W = H = canvas_px
    if filter_mode == "pair":
        canvas = Image.new("RGB", (W, H), bg_color)
    elif filter_mode == "global":
        base = fg_color

        def adjust(rgb, factor):
            return tuple(int(max(0, min(255, c * factor))) for c in rgb)

        bg_tone = adjust(base, 1.15)
        fg_tone = adjust(base, 0.5)
        canvas = Image.new("RGB", (W, H), bg_tone)
        fg_color = fg_tone
        bg_color = bg_tone
    else:
        raise ValueError("filter_mode 必须是 'pair' 或 'global'")

    draw = ImageDraw.Draw(canvas)

    # 掩码集合
    shape_canon = normalize_shape_token(shape)
    S = get_mask_coords(shape_canon, grid_size)
    total = len(S)

    # 计数
    nA = round(total * pctA / 100.0)
    nB = total - nA

    # 采样并打乱
    coords = S[:]
    random.shuffle(coords)
    A_coords = coords[:nA]
    B_coords = coords[nA:]

    # 网格→像素中心
    cell_w = W / grid_size
    cell_h = H / grid_size
    centers = {(i, j): (int((j + 0.5) * cell_w), int((i + 0.5) * cell_h))
               for i in range(grid_size) for j in range(grid_size)}

    # 字体与符号
    if symbol_key not in SYMBOL_PAIRS:
        raise ValueError("symbol_set must be one of XO/AB/SD/ZP")
    symA, symB = SYMBOL_PAIRS[symbol_key]
    font_px = max(12, int(W / (grid_size * font_k)))
    font = load_font(font_px, font_path=font_path)

    # 渲染
    for (i, j) in A_coords:
        draw_text_centered(draw, centers[(i, j)], symA, font, fill=fg_color)
    for (i, j) in B_coords:
        draw_text_centered(draw, centers[(i, j)], symB, font, fill=fg_color)

    meta = {
        "task": "Grid",
        "shape": shape_canon,
        "grid_size": grid_size,
        "pctA": pctA,  # symA 百分比（已含多数方）
        "symbol": symbol_key,
        "color": color_key,
        "nA": nA,
        "nB": nB,
        "total": total,
        "seed": seed
    }
    return canvas, meta


# =========================
# 5) 适配层：参数规范与生成
# =========================

def _coerce_params(params: dict) -> dict:
    """
    支持的入参（优先级从高到低）：
      - grid_level: 1..7  （首选）
        * majority: "symA" | "symB" | None(随机)   —— 仅在 grid_level 模式下使用
      - ratio / ratio_id  （兼容旧法；ratio 例如 "70%O"/"70%X"/"50/50"）
      - shape / shape_id(0..3)
      - symbol_set in {"XO","AB","SD","ZP"}
      - color_pair / color_pair_id(0..7)  —— "a_b" 形式均可，如 "navy_white"
      - filter_mode in {"pair","global"}  (默认 "pair")
      - grid_size (默认10), canvas_px(默认600), font_k(默认11.0), seed(可选), font_path(可选)
    """
    # shape（统一别名 → 中文规范）
    if "shape" in params:
        raw = str(params["shape"]).strip()
        shape = normalize_shape_token(raw)
        if shape not in SHAPES_CANON:
            raise ValueError(f"shape must be one of {SHAPES_CANON} (got {raw})")
    else:
        sid = int(params.get("shape_id", 0)) % len(SHAPES_CANON)
        shape = SHAPES_CANON[sid]

    # symbol
    symbol = params.get("symbol_set", "XO")
    if symbol not in SYMBOL_PAIRS:
        raise ValueError("symbol_set must be one of XO/AB/SD/ZP")

    # color pair
    if "color_pair" in params:
        color_pair = str(params["color_pair"]).replace("-", "_")
    else:
        cpid = int(params.get("color_pair_id", 0)) % len(SYMBOL_COLORS)
        color_pair = SYMBOL_COLORS[cpid]

    # 画法/尺寸
    filter_mode = params.get("filter_mode", "pair")
    if filter_mode not in ("pair", "global"):
        raise ValueError("filter_mode must be 'pair' or 'global'")
    grid_size = int(params.get("grid_size", 10))
    canvas_px = int(params.get("canvas_px", 600))
    font_k = float(params.get("font_k", 11.0))
    font_path = params.get("font_path", "")

    # 选择模式：优先 grid_level
    pctA = None
    majority = None
    if "grid_level" in params:
        lvl = int(params["grid_level"])
        if lvl not in GRID_LEVEL_TO_PCT:
            raise ValueError(f"grid_level must be 1..7")
        base_pct = GRID_LEVEL_TO_PCT[lvl]  # 针对“多数方”的百分比

        # 多数方
        majority = params.get("majority", None)  # "symA"/"symB"/None
        if majority not in (None, "symA", "symB"):
            raise ValueError('majority must be "symA", "symB", or None')

        # 先确定 seed（用于随机 majority）
        seed = params.get("seed", None)
        if seed is None:
            base = f"{lvl}|{shape}|{symbol}|{color_pair}|{filter_mode}|{grid_size}|{canvas_px}|{font_k}"
            # 可复现默认 seed
            seed = (abs(hash(base)) ^ 0xA5A5_5A5A) & 0xFFFFFFFF

        if majority is None:
            rnd = random.Random(int(seed))
            majority = rnd.choice(["symA", "symB"])

        pctA = base_pct if majority == "symA" else 100 - base_pct

    else:
        # 兼容旧 ratio / ratio_id
        if "ratio" in params:
            ratio = params["ratio"]
            if ratio not in PERCENTAGE_LEVELS:
                raise ValueError(f"ratio must be in {PERCENTAGE_LEVELS}")
        else:
            rid = int(params.get("ratio_id", 0)) % len(PERCENTAGE_LEVELS)
            ratio = PERCENTAGE_LEVELS[rid]

        # 将旧 ratio 解析为 pctA（约定 %O => symA 多；%X => symB 多）
        if ratio == "50/50":
            pctA = 50
        elif ratio.endswith("%O"):
            pctA = int(ratio[:-2])
        elif ratio.endswith("%X"):
            pctA = 100 - int(ratio[:-2])
        else:
            raise ValueError(f"Unknown ratio: {ratio}")

        # seed（旧模式也允许传）
        seed = params.get("seed", None)
        if seed is None:
            base = f"{ratio}|{shape}|{symbol}|{color_pair}|{filter_mode}|{grid_size}|{canvas_px}|{font_k}"
            seed = (abs(hash(base)) ^ 0xDEADBEEF) & 0xFFFFFFFF

    # 最终 seed
    seed = int(seed) if "seed" in locals() else int(params.get("seed", 0))

    return {
        "shape": shape,                    # 规范中文
        "symbol_set": symbol,
        "color_pair": color_pair,
        "filter_mode": filter_mode,
        "grid_size": grid_size,
        "canvas_px": canvas_px,
        "font_k": font_k,
        "font_path": font_path,
        "pctA": int(pctA),
        "seed": seed,
        # 仅当 grid_level 模式才返回以下两项（用于命名/记录）
        "grid_level": int(params.get("grid_level")) if "grid_level" in params else None,
        "majority": majority if "grid_level" in params else None,
    }



def _stable_filename(p: dict) -> str:
    """
    文件名同时兼容新老模式：
      新：Grid_{shapeEN}_N{N}_L{level}_maj{A/B}_{sym}_{color}_{seed}.png
      旧：Grid_{shapeEN}_N{N}_{ratioLike}_{sym}_{color}_{seed}.png
    """
    if p.get("grid_level") is not None:
        maj = "A" if p.get("majority") == "symA" else "B"
        tag = f"L{p['grid_level']}_maj{maj}"
    else:
        # 将 pctA 回写成类似 70%O/70%X/50/50 的标签仅用于命名（不建议再用）
        pctA = p["pctA"]
        if pctA == 50:
            tag = "50/50"
        elif pctA > 50:
            tag = f"{pctA}%O"
        else:
            tag = f"{100 - pctA}%X"

    # 中文规范 → 英文 slug（文件名更友好）
    shape_slug = SHAPE_EN_SLUG.get(normalize_shape_token(p['shape']), str(p['shape']))
    # 新的命名方式：只显示 symbol 颜色，背景统一为 gray
    color_display = p['color_pair'] if p['color_pair'] in SYMBOL_COLORS else p['color_pair']
    return f"Grid_{shape_slug}_N{p['grid_size']}_{tag}_{p['symbol_set']}_{color_display}_{p['seed']:08x}.png"


def generate(params: dict, out_dir: str) -> str:
    """
    统一入口：根据 params 生成单张网格 PNG 并返回路径。
    """
    p = _coerce_params(params)
    os.makedirs(out_dir, exist_ok=True)
    fname = _stable_filename(p)
    save_path = os.path.join(out_dir, fname)

    if os.path.exists(save_path):
        return save_path

    img, _meta = render_grid_image(
        grid_size=p["grid_size"],
        shape=p["shape"],        # 已是中文规范
        pctA=p["pctA"],
        symbol_key=p["symbol_set"],
        color_key=p["color_pair"],
        filter_mode=p["filter_mode"],
        canvas_px=p["canvas_px"],
        seed=p["seed"],
        font_k=p["font_k"],
        font_path=p["font_path"]
    )
    img.save(save_path)
    return save_path


# =========================
# 6) Ground Truth
# =========================

def xo_ground_truth(params: dict) -> Dict[str, str]:
    """
    返回：
      {"more_symbol": "symA" | "symB" | "equal", "symA": "...", "symB": "..."}
    """
    p = _coerce_params(params)
    symA, symB = SYMBOL_PAIRS[p["symbol_set"]]
    if p["pctA"] > 50:
        more = "symA"
    elif p["pctA"] < 50:
        more = "symB"
    else:
        more = "equal"
    return {"more_symbol": more, "symA": symA, "symB": symB}


# =========================
# 7) 批量生成（7×4×4×8 = 896）
# =========================

def append_csv(row: Dict, csv_path: Path):
    is_new = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "task", "shape", "grid_size", "grid_level", "majority",
            "pctA", "symbol", "color",
            "nA", "nB", "total", "seed", "image_path"
        ])
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def generate_all(
    output_dir: Path,
    log_csv: Path,
    grid_size: int = 10,
    canvas_px: int = 600,
    filter_mode: str = "pair",
    seed_base: int = 42,
    font_path: str = ""
):
    """
    每个组合（level×shape×symbol×color）只生成一张；
    多数方由组合+seed_base 确定的伪随机决定（可复现），因此总数固定 896 (7×4×4×8)。
    """
    output_dir = Path(output_dir)
    log_csv = Path(log_csv)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_csv.parent.mkdir(parents=True, exist_ok=True)

    for lvl in range(1, 8):
        base_pct = GRID_LEVEL_TO_PCT[lvl]
        for shape in SHAPES_CANON:  # 使用中文规范枚举
            for sym in SYMBOL_PAIRS.keys():
                for col in SYMBOL_COLORS:
                    # 组合可复现 seed & 多数方
                    base = f"{lvl}|{shape}|{sym}|{col}|{filter_mode}|{grid_size}|{canvas_px}"
                    seed = (seed_base * 1000003 + abs(hash(base))) & 0xFFFFFFFF
                    rnd = random.Random(seed)
                    majority = rnd.choice(["symA", "symB"])
                    pctA = base_pct if majority == "symA" else 100 - base_pct

                    img, meta = render_grid_image(
                        grid_size=grid_size,
                        shape=shape,
                        pctA=pctA,
                        symbol_key=sym,
                        color_key=col,
                        filter_mode=filter_mode,
                        canvas_px=canvas_px,
                        seed=seed,
                        font_k=11.0,
                        font_path=font_path
                    )

                    # 命名与保存
                    p_like = {
                        "shape": shape, "grid_size": grid_size, "grid_level": lvl,
                        "majority": "symA" if pctA > 50 else ("symB" if pctA < 50 else "equal"),
                        "symbol_set": sym, "color_pair": col, "seed": seed, "pctA": pctA
                    }
                    fn = _stable_filename(p_like)
                    save_path = output_dir / fn
                    img.save(save_path)

                    # CSV
                    meta_row = {
                        "task": "Grid",
                        "shape": shape,
                        "grid_size": grid_size,
                        "grid_level": lvl,
                        "majority": p_like["majority"],
                        "pctA": pctA,
                        "symbol": sym,
                        "color": col,
                        "nA": meta["nA"],
                        "nB": meta["nB"],
                        "total": meta["total"],
                        "seed": seed,
                        "image_path": str(save_path.as_posix())
                    }
                    append_csv(meta_row, log_csv)


# =========================
# 8) CLI
# =========================

def parse_args():
    ap = argparse.ArgumentParser(description="Grid（XO 网格）task生成器 · 新难度梯度版")
    ap.add_argument("--output_dir", type=str, default="stimuli/grid_levels")
    ap.add_argument("--log_csv", type=str, default="logs/grid_levels.csv")
    ap.add_argument("--grid_size", type=int, default=10)
    ap.add_argument("--canvas_px", type=int, default=600)
    ap.add_argument("--filter_mode", type=str, default="pair", choices=["pair", "global"])
    ap.add_argument("--seed_base", type=int, default=42)
    ap.add_argument("--font_path", type=str, default="")
    return ap.parse_args()


def main():
    args = parse_args()
    generate_all(
        output_dir=args.output_dir,
        log_csv=args.log_csv,
        grid_size=args.grid_size,
        canvas_px=args.canvas_px,
        filter_mode=args.filter_mode,
        seed_base=args.seed_base,
        font_path=args.font_path
    )
    print("Done. Generated 896 images and CSV:", args.log_csv)


if __name__ == "__main__":
    main()
