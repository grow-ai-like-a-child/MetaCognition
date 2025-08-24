# generators/color_shading.py
# -*- coding: utf-8 -*-
import os, io, hashlib, random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ===== 你的原始“常量”可放到这里，保留默认 =====
CANVAS   = (600, 600)             # 画布（改为600x600）
BG_VAL   = 80                     # 背景灰度 0-255
BASE_VAL = 180                    # 字体基准灰度 0-255
FONTSIZE = 50

# 布局与维度（与你的脚本一致）
LAYOUTS = ["LR", "UD", "DRUL", "DLUR"]  # 4

WORD_PAIRS = {  # 8
    1: ("DOG",  "CAT"),
    2: ("LION", "WOLF"),
    3: ("BIRD", "FISH"),
    4: ("TREE", "ROCK"),
    5: ("STAR", "MOON"),
    6: ("RAIN", "SNOW"),
    7: ("SHIP", "BOAT"),
    8: ("DEER", "BEAR"),
}

DELTA_L = [12, 24, 36, 48, 60, 72, 84]  # 7

FONT_STYLES = {  # 4
    1: {"label": "Regular",   "family": "DejaVu Sans", "weight": "normal", "style": "normal"},
    2: {"label": "Bold",      "family": "DejaVu Sans", "weight": "bold",   "style": "normal"},
    3: {"label": "Italic",    "family": "DejaVu Sans", "weight": "normal", "style": "italic"},
    4: {"label": "Monospace", "family": "monospace",   "weight": "normal", "style": "normal"},
}

# ========== 布局坐标（与你原代码一致） ==========
def positions_for_layout(layout):
    W, H = CANVAS
    if layout == "LR":
        return (W/4, H/2), (3*W/4, H/2)          # 左 vs 右
    if layout == "UD":
        return (W/2, 3*H/4), (W/2, H/4)          # 上 vs 下
    if layout == "DRUL":                          # 左下 vs 右上
        return (W/4, H/4), (3*W/4, 3*H/4)
    if layout == "DLUR":                          # 左上 vs 右下
        return (W/4, 3*H/4), (3*W/4, H/4)
    raise ValueError(f"Unknown layout: {layout}")

def render_two_words_grayscale(left_text, right_text, left_val, right_val,
                               layout="LR", fontsize=FONTSIZE, font_kwargs=None):
    W, H = CANVAS
    fig = plt.figure(figsize=(W/100, H/100), dpi=100)
    ax = plt.axes([0,0,1,1])
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis('off')
    ax.add_patch(plt.Rectangle((0,0), W, H, color=(BG_VAL/255, BG_VAL/255, BG_VAL/255)))

    left_pos, right_pos = positions_for_layout(layout)
    grayL = (left_val/255, left_val/255, left_val/255)
    grayR = (right_val/255, right_val/255, right_val/255)

    base_font = {"family": "DejaVu Sans", "weight": "normal", "style": "normal"}
    if font_kwargs:
        base_font.update(font_kwargs)

    ax.text(*left_pos, left_text, ha='center', va='center',
            fontsize=fontsize, color=grayL, **base_font)
    ax.text(*right_pos, right_text, ha='center', va='center',
            fontsize=fontsize, color=grayR, **base_font)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, transparent=False)
    plt.close(fig)
    buf.seek(0)
    return buf

# ========== 适配层：统一 generate() 接口 ==========
# 允许的“答案字典”（工程问法用到的 canonical tokens）
CANON_ANS = {
    "LR":   ("left", "right"),
    "UD":   ("top", "bottom"),
    "DRUL": ("bottom-left", "top-right"),
    "DLUR": ("top-left", "bottom-right"),
}

def _coerce_params(params: dict):
    """
    兼容工程与原脚本的入参：
      - layout:  "LR"/"UD"/"DRUL"/"DLUR"  或 layout_id ∈ {0..3}
      - word_pair_id ∈ {1..8}
      - delta_L ∈ {12,24,36,48,60,72,84}
      - fontstyle_id ∈ {1..4}
      - brighter_side 可选：'L'/'R'（留空则由 seed/散列决定，保证可复现）
      - seed 可选：决定未显式指定时更亮侧
      - 可覆盖：CANVAS/BG_VAL/BASE_VAL/FONTSIZE（如 img_size）
    """
    # layout
    if "layout" in params:
        layout = params["layout"]
        if layout not in LAYOUTS:
            raise ValueError(f"Unknown layout: {layout}")
    elif "layout_id" in params:
        lid = int(params["layout_id"]) % len(LAYOUTS)
        layout = LAYOUTS[lid]
    else:
        layout = "LR"

    # word pair
    if "word_pair_id" not in params:
        raise ValueError("Missing word_pair_id (1..8)")
    wp_id = int(params["word_pair_id"])
    if wp_id not in WORD_PAIRS:
        raise ValueError(f"word_pair_id out of range: {wp_id}")
    left_word, right_word = WORD_PAIRS[wp_id]

    # delta_L
    dL = params.get("delta_L")
    if dL is None:
        raise ValueError("Missing delta_L")
    dL = int(dL)
    if dL not in DELTA_L:
        raise ValueError(f"delta_L must be one of {DELTA_L}")

    # fontstyle
    fs_id = int(params.get("fontstyle_id", 1))
    if fs_id not in FONT_STYLES:
        raise ValueError(f"fontstyle_id out of range: {fs_id}")
    font = FONT_STYLES[fs_id]

    # geometry & tone overrides
    img_size  = int(params.get("img_size", CANVAS[0]))
    # 保持正方形
    canvas = (img_size, img_size)

    bg_val   = int(params.get("bg_gray",   BG_VAL))
    base_val = int(params.get("base_gray", BASE_VAL))
    fontsize = int(params.get("fontsize",  FONTSIZE))

    # brighter side
    bside = params.get("brighter_side", None)  # 'L' or 'R'
    seed  = params.get("seed", None)

    return {
        "layout": layout,
        "wp_id": wp_id,
        "left_word": left_word,
        "right_word": right_word,
        "delta_L": dL,
        "font": font,
        "fontstyle_id": fs_id,
        "canvas": canvas,
        "bg_val": bg_val,
        "base_val": base_val,
        "fontsize": fontsize,
        "brighter_side": bside,
        "seed": seed,
    }

def _derive_brighter_side(p: dict) -> str:
    """
    未显式给出 brighter_side 时，用确定性方式从参数派生：
    返回 'L' 或 'R'。
    """
    if p["brighter_side"] in ("L", "R"):
        return p["brighter_side"]

    # 用参数+seed 生成稳定散列，保证每个题唯一且可复现
    h = hashlib.blake2b(
        f"{p['layout']}-{p['wp_id']}-{p['delta_L']}-{p['fontstyle_id']}-{p['canvas']}-{p['bg_val']}-{p['base_val']}-{p['fontsize']}-{p.get('seed', '')}".encode("utf-8"),
        digest_size=2
    ).digest()
    # 偶数=左亮，奇数=右亮
    return "L" if (h[0] ^ h[1]) % 2 == 0 else "R"

def _stable_filename(p: dict, brighter_side: str) -> str:
    return (
        f"COL_gray_layout{p['layout']}"
        f"_wp{p['wp_id']:02d}_{p['left_word']}-{p['right_word']}"
        f"_dL{p['delta_L']:02d}_fs{p['fontstyle_id']}{p['font']['label']}"
        f"_b{brighter_side}.png"
    )

def _set_globals_from_canvas_and_bg(canvas, bg):
    global CANVAS, BG_VAL
    CANVAS = canvas
    BG_VAL = bg

def generate(params: dict, out_dir: str) -> str:
    """
    统一入口：根据 params 生成单张 PNG 并返回路径。
    入参示例：
      {
        "layout": "LR",            # 或 layout_id: 0..3
        "word_pair_id": 3,         # 1..8
        "delta_L": 36,             # 12..84 step 12
        "fontstyle_id": 2,         # 1..4
        "brighter_side": "L",      # 可选，不给则按散列确定
        "img_size": 256,           # 可选
        "bg_gray": 80,             # 可选
        "base_gray": 180,          # 可选
        "fontsize": 22,            # 可选
        "seed": 20250822           # 可选
      }
    """
    p = _coerce_params(params)
    bside = _derive_brighter_side(p)

    # 命名与缓存
    os.makedirs(out_dir, exist_ok=True)
    fname = _stable_filename(p, bside)
    save_path = os.path.join(out_dir, fname)
    if os.path.exists(save_path):
        return save_path

    # 计算左右亮度
    half = p["delta_L"] // 2
    if bside == "L":
        left_val, right_val = p["base_val"] + half, p["base_val"] - half
    else:
        left_val, right_val = p["base_val"] - half, p["base_val"] + half
    left_val  = int(np.clip(left_val,  0, 255))
    right_val = int(np.clip(right_val, 0, 255))

    # 覆盖全局画布与背景（仅对当前渲染有效）
    _set_globals_from_canvas_and_bg(p["canvas"], p["bg_val"])

    buf = render_two_words_grayscale(
        p["left_word"], p["right_word"], left_val, right_val,
        layout=p["layout"], fontsize=p["fontsize"],
        font_kwargs={"family": p["font"]["family"], "weight": p["font"]["weight"], "style": p["font"]["style"]}
    )
    with open(save_path, "wb") as f:
        f.write(buf.read())
    return save_path

# ========== Ground Truth（工程评测用） ==========
def color_ground_truth(params: dict) -> str:
    """
    返回 canonical 选项之一：
      - LR   -> "left"/"right"
      - UD   -> "top"/"bottom"
      - DRUL -> "bottom-left"/"top-right"
      - DLUR -> "top-left"/"bottom-right"
    """
    p = _coerce_params(params)
    bside = _derive_brighter_side(p)

    # 映射到 canonical token
    if p["layout"] == "LR":
        return "left" if bside == "L" else "right"
    if p["layout"] == "UD":
        # 注意：UD 中我们放字的位置是 (上, 下) 对应 left/right 槽位
        # positions_for_layout 给的是 (上, 下) → (left_slot, right_slot)
        return "top" if bside == "L" else "bottom"
    if p["layout"] == "DRUL":
        return "bottom-left" if bside == "L" else "top-right"
    if p["layout"] == "DLUR":
        return "top-left" if bside == "L" else "bottom-right"
    raise ValueError(f"Unknown layout: {p['layout']}")

# 可选：保留你原来的 batch main() 另名 bulk_main()，以便单独一键 896 生成
if __name__ == "__main__":
    pass
